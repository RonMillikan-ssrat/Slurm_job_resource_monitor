/*
jobmon.cpp

NOAA - 2025
Monitors CPU and GPU resource usage across all nodes of a running SLURM job.
Polls at a configurable interval and appends CSV rows to an output file.
Auto-terminates when the job is no longer active in the queue.

Compile:
  g++ -std=c++17 jobmon.cpp -o jobmon -lpthread

Usage:
  jobmon --job <job_id> [--interval <seconds>] [--out <csv_file>]
  jobmon -j <job_id> [-i <seconds>] [-o <csv_file>]

CSV columns:
  timestamp,job_id,node,cpu_pct,ram_used_mb,ram_total_mb,
  gpu_id,pid,gpu_mem_mib,gpu_util_pct,gpu_power_w,gpu_temp_c

The MIT License (MIT)
Copyright © 2025 NOAA
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <future>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <getopt.h>
#include <sys/wait.h>
#include <atomic>

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static std::atomic<bool> g_interrupted(false);

void sig_handler(int) {
    g_interrupted.store(true);
}

// ---------------------------------------------------------------------------
// Shell helpers
// ---------------------------------------------------------------------------

// Run a shell command and return stdout.  Throws on non-zero exit.
std::string exec(const std::string& cmd) {
    char buf[256];
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen() failed: " + cmd);
    try {
        while (fgets(buf, sizeof(buf), pipe))
            result += buf;
    } catch (...) {
        pclose(pipe);
        throw;
    }
    int status = pclose(pipe);
    if (WEXITSTATUS(status) != 0)
        throw std::runtime_error("Command failed (exit " +
                                 std::to_string(WEXITSTATUS(status)) +
                                 "): " + cmd +
                                 (result.empty() ? "" : "\n" + result));
    return result;
}

// Run a command on a remote node via SSH (BatchMode, short timeout).
std::string ssh_exec(const std::string& node, const std::string& remote_cmd) {
    std::string full = "ssh -oBatchMode=yes -oConnectTimeout=5 -oStrictHostKeyChecking=no "
                       + node + " \"" + remote_cmd + "\" 2>&1";
    return exec(full);
}

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

std::string timestamp_now() {
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm lm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&lm, "%Y:%m:%d:%H:%M:%S");
    return oss.str();
}

// ---------------------------------------------------------------------------
// SLURM helpers
// ---------------------------------------------------------------------------

// Returns true if the job is still present in the queue (any state).
bool job_is_active(const std::string& job_id) {
    try {
        std::string out = exec("squeue -j " + job_id + " -h 2>/dev/null");
        return !out.empty();
    } catch (...) {
        return false;
    }
}

// Expand a SLURM nodelist string into individual hostnames.
std::vector<std::string> expand_nodelist(const std::string& node_string) {
    std::vector<std::string> nodes;
    std::string out = exec("scontrol show hostnames " + node_string);
    std::istringstream ss(out);
    std::string node;
    while (std::getline(ss, node))
        if (!node.empty())
            nodes.push_back(node);
    return nodes;
}

// Get the list of nodes allocated to a job.
std::vector<std::string> get_job_nodes(const std::string& job_id) {
    std::string out = exec("scontrol show job " + job_id + " -o");
    std::regex rx(" NodeList=([^ ]+)");
    std::smatch m;
    if (!std::regex_search(out, m, rx) || m.size() < 2) {
        std::cerr << "Warning: NodeList not found for job " << job_id << std::endl;
        return {};
    }
    std::string node_str = m[1].str();
    if (node_str == "(null)")
        return {};
    return expand_nodelist(node_str);
}

// ---------------------------------------------------------------------------
// Per-node data structures
// ---------------------------------------------------------------------------

struct GpuRow {
    std::string gpu_id;
    std::string pid;          // may be empty
    std::string mem_mib;
    std::string util_pct;
    std::string power_w;
    std::string temp_c;
};

struct NodeSample {
    std::string node;
    bool        ok = false;
    std::string error_msg;

    // CPU / RAM (aggregate node)
    double      cpu_pct     = 0.0;
    long        ram_used_mb = 0;
    long        ram_total_mb= 0;

    // One entry per GPU
    std::vector<GpuRow> gpus;
};

// ---------------------------------------------------------------------------
// CPU/RAM parsing  (/proc/stat  and  /proc/meminfo via SSH)
// ---------------------------------------------------------------------------

// We do two reads of /proc/stat 500 ms apart to compute CPU %.
// Format of /proc/stat first line:
//   cpu  user nice system idle iowait irq softirq steal guest guest_nice
struct CpuTicks {
    long long idle = 0, total = 0;
};

CpuTicks parse_cpu_stat(const std::string& stat_line) {
    // "cpu  12345 0 6789 ..."
    std::istringstream ss(stat_line);
    std::string label;
    ss >> label;
    CpuTicks ct;
    long long v;
    int idx = 0;
    while (ss >> v) {
        ct.total += v;
        if (idx == 3) ct.idle = v; // 4th field is idle
        ++idx;
    }
    return ct;
}

// Returns aggregate cpu % and fills ram fields.
// remote_cmd must return:
//   LINE1: first line of /proc/stat
//   LINE2: first line of /proc/stat (500 ms later)
//   LINE3+: /proc/meminfo
void parse_node_resources(const std::string& raw,
                          double& cpu_pct,
                          long& ram_used_mb,
                          long& ram_total_mb) {
    std::istringstream ss(raw);
    std::string line1, line2;
    std::getline(ss, line1);  // first stat snapshot
    std::getline(ss, line2);  // second stat snapshot

    CpuTicks t1 = parse_cpu_stat(line1);
    CpuTicks t2 = parse_cpu_stat(line2);

    long long dtotal = t2.total - t1.total;
    long long didle  = t2.idle  - t1.idle;
    cpu_pct = (dtotal > 0) ? 100.0 * (1.0 - (double)didle / dtotal) : 0.0;

    // /proc/meminfo
    long mem_total_kb = 0, mem_avail_kb = 0;
    std::regex mem_total_rx(R"(^MemTotal:\s+(\d+))");
    std::regex mem_avail_rx(R"(^MemAvailable:\s+(\d+))");
    std::smatch m;
    std::string mline;
    while (std::getline(ss, mline)) {
        if (std::regex_search(mline, m, mem_total_rx))
            mem_total_kb = std::stol(m[1].str());
        else if (std::regex_search(mline, m, mem_avail_rx))
            mem_avail_kb = std::stol(m[1].str());
    }
    ram_total_mb = mem_total_kb / 1024;
    ram_used_mb  = (mem_total_kb - mem_avail_kb) / 1024;
}

// ---------------------------------------------------------------------------
// GPU parsing  (nvidia-smi output)
// ---------------------------------------------------------------------------

std::vector<GpuRow> parse_gpu(const std::string& smi_out) {
    std::vector<GpuRow> rows;

    // Build pid->gpu_id map from Processes table
    // Line pattern inside processes section:
    //   |  GPU-idx   GI   CI        PID   Type   Process name    GPU Memory  |
    //   |    0        0    0      12345      C   python               4321MiB |
    std::unordered_map<std::string, std::string> gpu_for_pid;
    std::regex proc_rx(R"(^\|\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+C\b)",
                       std::regex::multiline);
    for (std::sregex_iterator it(smi_out.begin(), smi_out.end(), proc_rx), end;
         it != end; ++it) {
        const std::smatch& pm = *it;
        gpu_for_pid[pm[1].str()] = pm[2].str(); // gpu_idx -> pid
    }

    // Parse per-GPU stats block
    // We step through lines; track current gpu index, then match the stats line.
    std::regex gpu_idx_rx(R"(^\|\s*(\d+)\s)");
    std::regex stats_rx(
        R"(^\|\s*\S+\s+(\d+)C\s+P\d+\s+(\d+)W\s*/\s*\d+W)"   // temp, power
        R"(\s*\|\s*(\d+)MiB\s*/\s*\d+MiB)"                     // mem used
        R"(\s*\|\s*(\d+)%\s+Default)"                           // util
    );

    std::string cur_gpu;
    std::istringstream ss(smi_out);
    std::string line;
    while (std::getline(ss, line)) {
        std::smatch gm;
        if (std::regex_search(line, gm, gpu_idx_rx)) {
            cur_gpu = gm[1].str();
        } else if (!cur_gpu.empty() && std::regex_search(line, gm, stats_rx)) {
            GpuRow row;
            row.gpu_id  = cur_gpu;
            row.temp_c  = gm[1].str();
            row.power_w = gm[2].str();
            row.mem_mib = gm[3].str();
            row.util_pct= gm[4].str();
            row.pid     = gpu_for_pid.count(cur_gpu) ? gpu_for_pid[cur_gpu] : "";
            rows.push_back(row);
            cur_gpu.clear();
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Per-node collection  (runs in its own thread)
// ---------------------------------------------------------------------------

NodeSample collect_node(const std::string& node) {
    NodeSample s;
    s.node = node;

    // --- CPU / RAM ---
    // Shell snippet: print /proc/stat, sleep 0.5, print /proc/stat again,
    // then dump /proc/meminfo.
    const std::string res_cmd =
        "grep -m1 '^cpu ' /proc/stat; sleep 0.5; "
        "grep -m1 '^cpu ' /proc/stat; "
        "cat /proc/meminfo";

    try {
        std::string res_out = ssh_exec(node, res_cmd);
        parse_node_resources(res_out,
                             s.cpu_pct,
                             s.ram_used_mb,
                             s.ram_total_mb);
    } catch (const std::exception& e) {
        s.ok = false;
        s.error_msg = std::string("CPU/RAM error: ") + e.what();
        return s;
    }

    // --- GPU ---
    try {
        std::string smi_out = ssh_exec(node, "nvidia-smi");
        s.gpus = parse_gpu(smi_out);
    } catch (const std::exception& e) {
        // GPU may not be available on all nodes; treat as warning, not fatal.
        std::cerr << "Warning [" << node << "]: GPU query failed: " << e.what() << std::endl;
    }

    s.ok = true;
    return s;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

// Header written once at file open.
static const char* CSV_HEADER =
    "timestamp,job_id,node,"
    "cpu_pct,ram_used_mb,ram_total_mb,"
    "gpu_id,pid,gpu_mem_mib,gpu_util_pct,gpu_power_w,gpu_temp_c\n";

void write_samples(std::ofstream& out,
                   const std::string& job_id,
                   const std::string& ts,
                   const std::vector<NodeSample>& samples) {
    for (const auto& s : samples) {
        if (!s.ok) {
            std::cerr << "Skipping node " << s.node
                      << " (" << s.error_msg << ")" << std::endl;
            continue;
        }

        if (s.gpus.empty()) {
            // Write one row with empty GPU fields
            out << ts << ","
                << job_id << ","
                << s.node << ","
                << std::fixed << std::setprecision(1) << s.cpu_pct << ","
                << s.ram_used_mb << ","
                << s.ram_total_mb << ","
                << ",,,,," << "\n";
        } else {
            for (const auto& g : s.gpus) {
                out << ts << ","
                    << job_id << ","
                    << s.node << ","
                    << std::fixed << std::setprecision(1) << s.cpu_pct << ","
                    << s.ram_used_mb << ","
                    << s.ram_total_mb << ","
                    << g.gpu_id << ","
                    << g.pid << ","
                    << g.mem_mib << ","
                    << g.util_pct << ","
                    << g.power_w << ","
                    << g.temp_c << "\n";
            }
        }
    }
    out.flush();
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " -j <job_id> [-i <interval_sec>] [-o <output.csv>] [-h]\n\n"
              << "Options:\n"
              << "  -j, --job      <id>    SLURM job ID to monitor (required)\n"
              << "  -i, --interval <sec>   Polling interval in seconds (default: 30)\n"
              << "  -o, --out      <file>  Output CSV path (default: jobmon_<job_id>.csv)\n"
              << "  -h, --help             Show this message\n\n"
              << "CSV columns:\n"
              << "  " << CSV_HEADER;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    std::string job_id;
    int         interval_sec = 30;
    std::string out_path;

    static struct option long_opts[] = {
        {"job",      required_argument, 0, 'j'},
        {"interval", required_argument, 0, 'i'},
        {"out",      required_argument, 0, 'o'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, li = 0;
    while ((opt = getopt_long(argc, argv, "j:i:o:h", long_opts, &li)) != -1) {
        switch (opt) {
            case 'j': job_id       = optarg; break;
            case 'i': interval_sec = std::stoi(optarg); break;
            case 'o': out_path     = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
            default:
                std::cerr << "Try '" << argv[0] << " --help'\n";
                return 1;
        }
    }

    if (job_id.empty()) {
        std::cerr << "Error: --job <job_id> is required.\n";
        print_usage(argv[0]);
        return 1;
    }
    if (interval_sec < 1) {
        std::cerr << "Error: interval must be >= 1 second.\n";
        return 1;
    }
    if (out_path.empty())
        out_path = "jobmon_" + job_id + ".csv";

    // Open CSV (append so re-runs don't clobber existing data)
    bool write_header = false;
    {
        std::ifstream test(out_path);
        write_header = !test.good(); // write header only for new files
    }
    std::ofstream csv(out_path, std::ios::app);
    if (!csv.is_open()) {
        std::cerr << "Error: cannot open output file: " << out_path << "\n";
        return 1;
    }
    if (write_header)
        csv << CSV_HEADER;

    std::cout << "jobmon: monitoring SLURM job " << job_id
              << " every " << interval_sec << "s -> " << out_path << "\n"
              << "Press Ctrl+C or wait for job to finish.\n";

    // --- Main polling loop ---
    while (!g_interrupted.load()) {
        if (!job_is_active(job_id)) {
            std::cout << "Job " << job_id << " is no longer active. Exiting.\n";
            break;
        }

        std::vector<std::string> nodes;
        try {
            nodes = get_job_nodes(job_id);
        } catch (const std::exception& e) {
            std::cerr << "Warning: could not get node list: " << e.what() << "\n";
        }

        if (nodes.empty()) {
            std::cout << "No nodes allocated yet for job " << job_id
                      << ". Waiting...\n";
        } else {
            // Collect from all nodes concurrently
            std::vector<std::future<NodeSample>> futures;
            futures.reserve(nodes.size());
            for (const auto& node : nodes)
                futures.push_back(
                    std::async(std::launch::async, collect_node, node));

            std::vector<NodeSample> samples;
            samples.reserve(futures.size());
            std::string ts = timestamp_now();
            for (auto& f : futures) {
                try {
                    samples.push_back(f.get());
                } catch (const std::exception& e) {
                    std::cerr << "Error collecting sample: " << e.what() << "\n";
                }
            }

            write_samples(csv, job_id, ts, samples);
            std::cout << ts << " sampled " << samples.size()
                      << " node(s).\n";
        }

        // Sleep interval_sec, but wake early if interrupted
        for (int s = 0; s < interval_sec && !g_interrupted.load(); ++s)
            std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    csv.close();
    std::cout << "jobmon: done. Output: " << out_path << "\n";
    return 0;
}
