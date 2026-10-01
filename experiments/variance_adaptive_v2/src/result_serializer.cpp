// result_serializer.cpp — Implementation of all JSON/CSV output functions.
//
// JSON is written manually (no external library) to keep the build simple.
// The format is human-readable but not pretty-printed to avoid bloat for
// large trace files.

#include "result_serializer.h"
#include <cstdio>
#include <cstring>
#include <direct.h>    // _mkdir on Windows
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// ---------------------------------------------------------------------------
// Directory creation (Windows)
// ---------------------------------------------------------------------------
static void make_dir(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR))
        return; // already exists
    // Try to create; ignore "already exists" error
    _mkdir(path.c_str());
}

void ResultSerializer::ensure_dir(const std::string& path) {
    // Walk the path and create each component
    std::string cur;
    for (char c : path) {
        if (c == '/' || c == '\\') {
            if (!cur.empty()) make_dir(cur);
        }
        cur += c;
    }
    if (!cur.empty()) make_dir(cur);
}

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------
std::string ResultSerializer::csv_str(const std::string& s) {
    if (s.find(',') == std::string::npos &&
        s.find('"') == std::string::npos &&
        s.find('\n') == std::string::npos)
        return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    out += '"';
    return out;
}

static std::string fmt(double v, int prec = 6) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
ResultSerializer::ResultSerializer(const std::string& out_dir)
    : out_dir_(out_dir)
{
    ensure_dir(out_dir_);

    // Open trace CSV
    std::string trace_path = out_dir_ + "/scheduler_trace.csv";
    trace_csv_.open(trace_path);
    if (!trace_csv_)
        throw std::runtime_error("Cannot open " + trace_path);

    trace_csv_ << "rep,arm,decision_idx,task_start,task_end,chunk_size,"
                  "window_start_task,window_end_task,window_n,"
                  "window_mean,window_variance,phase_id_at_decision,"
                  "t_kernel_ms,t_total_chunk_ms,"
                  "observation_lag_tasks,action_lag_tasks,is_response_decision\n";

    // Open raw results CSV
    std::string raw_path = out_dir_ + "/raw_results.csv";
    raw_csv_.open(raw_path);
    if (!raw_csv_)
        throw std::runtime_error("Cannot open " + raw_path);

    raw_csv_ << "rep,workload_seed,"
                "t_total_adaptive_ms,t_total_static_large_ms,t_total_static_matched_ms,"
                "t_total_inverted_adaptive_ms,"
                "mean_adaptive_chunk,n_chunks_adaptive,"
                "n_chunks_static_large,n_chunks_static_matched,n_chunks_inverted,arm_order\n";
}

ResultSerializer::~ResultSerializer() {
    if (trace_csv_.is_open()) trace_csv_.close();
    if (raw_csv_.is_open())   raw_csv_.close();
}

// ---------------------------------------------------------------------------
// write_config
// ---------------------------------------------------------------------------
void ResultSerializer::write_config(const ExperimentConfig& cfg,
                                    const std::vector<PhaseInfo>& phases)
{
    std::ofstream f(out_dir_ + "/config.json");
    if (!f) throw std::runtime_error("Cannot open config.json");

    f << "{\n";
    f << "  \"experiment_name\": \"" << cfg.experiment_name << "\",\n";
    f << "  \"total_tasks\": "       << cfg.total_tasks       << ",\n";
    f << "  \"cost_multiplier\": "   << cfg.cost_multiplier   << ",\n";
    f << "  \"workload_seed_base\": "<< cfg.workload_seed      << ",\n";
    f << "  \"min_chunk\": "         << cfg.min_chunk          << ",\n";
    f << "  \"max_chunk\": "         << cfg.max_chunk          << ",\n";
    f << "  \"chunk_step\": "        << cfg.chunk_step         << ",\n";
    f << "  \"static_large_chunk\": "<< cfg.static_large_chunk << ",\n";
    f << "  \"threads_per_block\": " << cfg.threads_per_block  << ",\n";
    f << "  \"window_size\": "       << cfg.window_size        << ",\n";
    f << "  \"var_low_thresh\": "    << fmt(cfg.var_low_thresh) << ",\n";
    f << "  \"var_high_thresh\": "   << fmt(cfg.var_high_thresh)<< ",\n";
    f << "  \"response_fraction\": " << fmt(cfg.response_fraction) << ",\n";
    f << "  \"num_warmup_reps\": "   << cfg.num_warmup_reps    << ",\n";
    f << "  \"num_final_reps\": "    << cfg.num_final_reps     << ",\n";
    f << "  \"inter_arm_sleep_ms\": "<< cfg.inter_arm_sleep_ms << ",\n";
    f << "  \"alpha\": "             << fmt(cfg.alpha)         << ",\n";
    f << "  \"min_effect_fraction\": "<< fmt(cfg.min_effect_fraction) << ",\n";
    f << "  \"statistical_test\": \"Wilcoxon signed-rank (two-tailed, paired)\",\n";
    f << "  \"primary_comparison\": \"STATIC-MATCHED vs ADAPTIVE\",\n";
    f << "  \"secondary_comparison\": \"STATIC-LARGE vs ADAPTIVE\",\n";
    f << "  \"phases\": [\n";
    for (int i = 0; i < (int)phases.size(); ++i) {
        const auto& p = phases[i];
        f << "    {\"phase_id\":" << p.phase_id
          << ", \"name\":\"" << p.name << "\""
          << ", \"start_task\":" << p.start_task
          << ", \"end_task\":" << p.end_task
          << ", \"distribution\":\"" << p.distribution_desc << "\""
          << ", \"actual_mean\":" << fmt(p.actual_mean)
          << ", \"actual_variance\":" << fmt(p.actual_variance)
          << ", \"actual_min\":" << p.actual_min
          << ", \"actual_max\":" << p.actual_max
          << "}";
        if (i + 1 < (int)phases.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
}

// ---------------------------------------------------------------------------
// write_workload
// ---------------------------------------------------------------------------
void ResultSerializer::write_workload(const std::vector<int>& task_costs,
                                      const std::vector<PhaseInfo>& phases)
{
    std::ofstream f(out_dir_ + "/workload.csv");
    if (!f) throw std::runtime_error("Cannot open workload.csv");
    f << "task_index,task_cost,phase_id\n";
    for (int i = 0; i < (int)task_costs.size(); ++i) {
        int pid = -1;
        for (const auto& p : phases)
            if (i >= p.start_task && i < p.end_task) { pid = p.phase_id; break; }
        f << i << "," << task_costs[i] << "," << pid << "\n";
    }
}

// ---------------------------------------------------------------------------
// append_trace_row
// ---------------------------------------------------------------------------
void ResultSerializer::append_trace_row(int rep,
                                        const SchedulerDecision& d,
                                        const std::string& arm_name)
{
    trace_csv_
        << rep << ","
        << csv_str(arm_name) << ","
        << d.decision_idx << ","
        << d.task_start << ","
        << d.task_end << ","
        << d.chunk_size << ","
        << d.window_start_task << ","
        << d.window_end_task << ","
        << d.window_n << ","
        << fmt(d.window_mean) << ","
        << fmt(d.window_variance) << ","
        << d.phase_id_at_decision << ","
        << fmt(d.t_kernel_ms, 4) << ","
        << fmt(d.t_total_chunk_ms, 4) << ","
        << d.observation_lag_tasks << ","
        << d.action_lag_tasks << ","
        << (d.is_response_decision ? 1 : 0) << "\n";
}

// ---------------------------------------------------------------------------
// append_rep_result
// ---------------------------------------------------------------------------
void ResultSerializer::append_rep_result(int rep,
                                          unsigned int workload_seed,
                                          double t_total_adaptive_ms,
                                          double t_total_static_large_ms,
                                          double t_total_static_matched_ms,
                                          double t_total_inverted_ms,
                                          int mean_adaptive_chunk,
                                          int n_chunks_adaptive,
                                          int n_chunks_static_large,
                                          int n_chunks_static_matched,
                                          int n_chunks_inverted,
                                          const std::string& arm_order)
{
    raw_csv_
        << rep << ","
        << workload_seed << ","
        << fmt(t_total_adaptive_ms, 4) << ","
        << fmt(t_total_static_large_ms, 4) << ","
        << fmt(t_total_static_matched_ms, 4) << ","
        << fmt(t_total_inverted_ms, 4) << ","
        << mean_adaptive_chunk << ","
        << n_chunks_adaptive << ","
        << n_chunks_static_large << ","
        << n_chunks_static_matched << ","
        << n_chunks_inverted << ","
        << csv_str(arm_order) << "\n";
    raw_csv_.flush();
}

// ---------------------------------------------------------------------------
// write_statistical_results
// ---------------------------------------------------------------------------
static void write_arm_stats(std::ofstream& f, const ArmStats& s, bool comma) {
    f << "  \"" << s.arm_name << "\": {"
      << "\"n\":" << s.n
      << ", \"mean_ms\":" << fmt(s.mean_ms)
      << ", \"std_ms\":" << fmt(s.std_ms)
      << ", \"min_ms\":" << fmt(s.min_ms)
      << ", \"max_ms\":" << fmt(s.max_ms)
      << "}";
    if (comma) f << ",";
    f << "\n";
}

static void write_wilcoxon(std::ofstream& f, const WilcoxonResult& w,
                            const std::string& key, bool comma) {
    f << "  \"" << key << "\": {\n";
    f << "    \"arm_a\": \"" << w.arm_a << "\",\n";
    f << "    \"arm_b\": \"" << w.arm_b << "\",\n";
    f << "    \"n\": " << w.n << ",\n";
    f << "    \"W_stat\": " << fmt(w.W_stat) << ",\n";
    f << "    \"Z_score\": " << fmt(w.Z_score) << ",\n";
    f << "    \"p_value\": " << fmt(w.p_value, 8) << ",\n";
    f << "    \"effect_size_r\": " << fmt(w.effect_size_r) << ",\n";
    f << "    \"mean_diff_ms\": " << fmt(w.mean_diff_ms) << ",\n";
    f << "    \"std_diff_ms\": " << fmt(w.std_diff_ms) << ",\n";
    f << "    \"relative_pct\": " << fmt(w.relative_pct, 4) << ",\n";
    f << "    \"ci_lower_95\": " << fmt(w.ci_lower_95) << ",\n";
    f << "    \"ci_upper_95\": " << fmt(w.ci_upper_95) << ",\n";
    f << "    \"is_significant\": " << (w.is_significant ? "true" : "false") << ",\n";
    f << "    \"meets_effect_threshold\": "
      << (w.meets_effect_threshold ? "true" : "false") << ",\n";
    f << "    \"is_improvement\": " << (w.is_improvement ? "true" : "false") << ",\n";
    f << "    \"interpretation\": \"" << w.interpretation << "\"";
    if (!w.caution.empty())
        f << ",\n    \"caution\": \"" << w.caution << "\"";
    f << "\n  }";
    if (comma) f << ",";
    f << "\n";
}

void ResultSerializer::write_statistical_results(
    const ArmStats& stats_adaptive,
    const ArmStats& stats_static_large,
    const ArmStats& stats_static_matched,
    const ArmStats& stats_inverted,
    const WilcoxonResult& primary_test,
    const WilcoxonResult& secondary_test,
    const WilcoxonResult& tertiary_test)
{
    std::ofstream f(out_dir_ + "/statistical_results.json");
    if (!f) throw std::runtime_error("Cannot open statistical_results.json");
    f << "{\n";
    f << "  \"note\": \"Primary: STATIC-MATCHED vs ADAPTIVE. "
         "Secondary: STATIC-LARGE vs ADAPTIVE. "
         "Tertiary: INVERTED-ADAPTIVE vs ADAPTIVE (mechanism test).\",\n";
    write_arm_stats(f, stats_adaptive,       true);
    write_arm_stats(f, stats_static_large,   true);
    write_arm_stats(f, stats_static_matched, true);
    write_arm_stats(f, stats_inverted,       true);
    write_wilcoxon(f, primary_test,   "primary_test_matched_vs_adaptive",   true);
    write_wilcoxon(f, secondary_test, "secondary_test_large_vs_adaptive",   true);
    write_wilcoxon(f, tertiary_test,  "tertiary_test_inverted_vs_adaptive", false);
    f << "}\n";
}

// ---------------------------------------------------------------------------
// write_detection_lag_results
// ---------------------------------------------------------------------------
void ResultSerializer::write_detection_lag_results(
    const std::vector<std::vector<DetectionLagResult>>& all_rep_lags,
    const std::vector<PhaseInfo>& phases)
{
    std::ofstream f(out_dir_ + "/detection_lag_results.json");
    if (!f) throw std::runtime_error("Cannot open detection_lag_results.json");
    f << "{\n";
    f << "  \"note\": \"observation_lag bounded by WINDOW_SIZE; "
         "action_lag >= observation_lag due to in-flight chunk delay.\",\n";
    f << "  \"transitions\": [\n";

    // Average across reps for each transition index
    if (!all_rep_lags.empty() && !all_rep_lags[0].empty()) {
        int n_trans = (int)all_rep_lags[0].size();
        for (int t = 0; t < n_trans; ++t) {
            int obs_sum = 0, act_sum = 0, obs_valid = 0, act_valid = 0;
            for (const auto& rep_lags : all_rep_lags) {
                if (t < (int)rep_lags.size()) {
                    if (rep_lags[t].observation_lag_tasks >= 0) {
                        obs_sum += rep_lags[t].observation_lag_tasks;
                        ++obs_valid;
                    }
                    if (rep_lags[t].action_lag_tasks >= 0) {
                        act_sum += rep_lags[t].action_lag_tasks;
                        ++act_valid;
                    }
                }
            }
            const auto& ref = all_rep_lags[0][t];
            f << "    {\"transition_index\":" << t
              << ", \"from_phase\":" << ref.from_phase
              << ", \"to_phase\":" << ref.to_phase
              << ", \"boundary_task\":" << ref.boundary_task
              << ", \"new_regime_high_variance\":"
              << (ref.new_regime_high_variance ? "true" : "false")
              << ", \"mean_observation_lag_tasks\":"
              << (obs_valid > 0 ? (double)obs_sum / obs_valid : -1.0)
              << ", \"mean_action_lag_tasks\":"
              << (act_valid > 0 ? (double)act_sum / act_valid : -1.0)
              << ", \"reps_with_detection\":" << obs_valid
              << "}";
            if (t + 1 < n_trans) f << ",";
            f << "\n";
        }
    }
    f << "  ]\n}\n";
}

// ---------------------------------------------------------------------------
// write_calibration (static)
// ---------------------------------------------------------------------------
void ResultSerializer::write_calibration(
    const std::string& out_dir,
    int chosen_multiplier,
    double overhead_ms,
    double var_low_thresh,
    double var_high_thresh,
    const std::vector<std::pair<int,double>>& sweep)
{
    ensure_dir(out_dir);
    std::ofstream f(out_dir + "/calibration.json");
    if (!f) throw std::runtime_error("Cannot open calibration.json");
    f << "{\n";
    f << "  \"chosen_cost_multiplier\": " << chosen_multiplier << ",\n";
    f << "  \"launch_overhead_ms\": " << fmt(overhead_ms) << ",\n";
    f << "  \"var_low_thresh_set\": " << fmt(var_low_thresh) << ",\n";
    f << "  \"var_high_thresh_set\": " << fmt(var_high_thresh) << ",\n";
    f << "  \"selection_criterion\": \"smallest multiplier where T_kernel >= 10x launch_overhead\",\n";
    f << "  \"multiplier_sweep\": [\n";
    for (int i = 0; i < (int)sweep.size(); ++i) {
        f << "    {\"multiplier\":" << sweep[i].first
          << ", \"t_kernel_ms\":" << fmt(sweep[i].second) << "}";
        if (i + 1 < (int)sweep.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
}
