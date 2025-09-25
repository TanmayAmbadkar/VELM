#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace flowstar;

// ReLU activation function
double my_relu(double v)
{
    return max(v, 0.0);
}

// Custom safety loss function for the Hopper environment.
// Calculates a loss based on the state variable x6 (joint angle velocity).
// The safe condition is -0.37315 <= x6 <= 0.37315.
double safe_loss_hopper(double x6_inf, double x6_sup)
{
    // Loss is how much the interval [x6_inf, x6_sup] violates the safe bounds.
    double loss_upper = my_relu(x6_sup - 0.37315);
    double loss_lower = my_relu(-0.37315 - x6_inf);
    return max(loss_upper, loss_lower);
}

int main(int argc, char *argv[])
{
    // --- Argument Parsing ---
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <controller_base_path> [--plot | --safe_sets]" << endl;
        return 1;
    }
    bool plot = false;
    bool print_safe_sets = false;
    if (argc >= 3)
    {
        if (string(argv[2]) == "--plot")
        {
            plot = true;
        }
        else if (string(argv[2]) == "--safe_sets")
        {
            print_safe_sets = true;
        }
    }
    string benchmark_name = "hopper";

    // --- Variable Declaration ---
    // 11 state variables + 3 action variables = 14 total
    unsigned int numVars = 14;
    intervalNumPrecision = 200;
    Variables vars;
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int x5_id = vars.declareVar("x5");
    int x6_id = vars.declareVar("x6");
    int x7_id = vars.declareVar("x7");
    int x8_id = vars.declareVar("x8");
    int x9_id = vars.declareVar("x9");
    int x10_id = vars.declareVar("x10");
    int x11_id = vars.declareVar("x11");
    int u1_id = vars.declareVar("u1"); // Action 1
    int u2_id = vars.declareVar("u2"); // Action 2
    int u3_id = vars.declareVar("u3"); // Action 3

    // --- Load Learned Dynamics ---
    ifstream dynamics_file("./results/learned_dynamics/Marvelgymnasium_hopper-v1/model.txt");
    if (!dynamics_file.is_open()) {
        cerr << "Error: Could not open dynamics file." << endl;
        return 1;
    }
    vector<string> dynamics_str;
    string equation;
    for (int i = 0; i < 11; ++i)
    {
        dynamics_file >> equation;
        dynamics_str.push_back(equation);
    }
    // The 3 action variables have no dynamics of their own.
    dynamics_str.push_back("0"); // u1
    dynamics_str.push_back("0"); // u2
    dynamics_str.push_back("0"); // u3
    DDE<Real> dynamics(dynamics_str, vars);

    // --- Reachability Settings ---
    Computational_Setting setting(vars);
    unsigned int order = 4;
    setting.setFixedStepsize(0.01, order);
    setting.setCutoffThreshold(1e-8);
    setting.printOff();

    // --- Initial Set Definition (matches Python environment's init_space) ---
    int steps = 200;
    vector<Interval> X0;
    X0.push_back(Interval(-0.005, 0.005)); // x1: pos_x
    X0.push_back(Interval(1.2, 1.3));     // x2: pos_z (height)
    X0.push_back(Interval(-0.05, 0.05));  // x3: ang_y
    X0.push_back(Interval(-0.05, 0.05));  // x4: ang_z
    X0.push_back(Interval(-0.05, 0.05));  // x5: ang_x
    X0.push_back(Interval(-0.05, 0.05));  // x6: vel_x
    X0.push_back(Interval(-0.05, 0.05));  // x7: vel_z
    X0.push_back(Interval(-0.05, 0.05));  // x8: vel_ang_y
    X0.push_back(Interval(-0.05, 0.05));  // x9: vel_ang_z
    X0.push_back(Interval(-0.05, 0.05));  // x10: vel_ang_x
    X0.push_back(Interval(-0.05, 0.05));  // x11: vel_root_x
    X0.push_back(Interval(0));            // u1: action 1 starts at 0
    X0.push_back(Interval(0));            // u2: action 2 starts at 0
    X0.push_back(Interval(0));            // u3: action 3 starts at 0

    Flowpipe initial_set(X0);
    Symbolic_Remainder symbolic_remainder(initial_set, 100);
    vector<Constraint> safeSet; // No explicit safe set constraints needed
    Result_of_Reachability result;

    unsigned int bernstein_order = 2;
    unsigned int partition_num = 200;
    string controller_base = string(argv[1]);
    int interval = 40; // How often to load a new controller file
    NeuralNetwork *nn = nullptr;
    double total_safe_loss = 0.0;

    // --- Main Reachability Loop ---
    for (int iter = 0; iter < steps; ++iter)
    {
        // Load the controller neural network at specified intervals
        if (iter % interval == 0)
        {
            if (nn != nullptr) delete nn;
            nn = new NeuralNetwork(controller_base + "_" + to_string(iter));
        }

        // Prepare the input for the neural network (state variables)
        TaylorModelVec<Real> tmv_input;
        for (int i = 0; i < 11; ++i) {
            tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
        }
        
        // Propagate the state through the controller to get the action
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
        polar_setting.set_num_threads(-1);
        TaylorModelVec<Real> tmv_output;
        nn->get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

        // Assign the controller outputs to the action variables
        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2];

        // Compute the reachable set for the next step using the learned dynamics
        dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);

        if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
        {
            initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
            result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
            
            // Calculate and accumulate the safety loss
            total_safe_loss += safe_loss_hopper(inter_box[5].inf(), inter_box[5].sup());

            if (print_safe_sets)
            {
                for(size_t i = 0; i < 11; ++i) {
                    cout << inter_box[i].inf() << " " << inter_box[i].sup() << " ";
                }
                cout << "\n";
            }
        }
        else
        {
            printf("Terminated due to too large overestimation.\n");
            return 1;
        }
    }
    delete nn;

    // --- Plotting ---
    if (plot)
    {
        result.transformToTaylorModels(setting);
        Plot_Setting plot_setting(vars);
        // Plot horizontal position (x1) vs height (x2)
        plot_setting.setOutputDims("x1", "x2");

        int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        if (mkres < 0 && errno != EEXIST)
        {
            printf("Can not create the directory for images.\n");
            exit(1);
        }
        plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name, result.tmv_flowpipes, setting);
    }

    cout << total_safe_loss << endl;
    return 0;
}
