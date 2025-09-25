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

// Custom safety loss function for the cheetah environment.
// Calculates a loss based on the state variable x10 (a joint velocity).
// The safe condition is -2.8795 <= x10 <= 2.8795.
double safe_loss_cheetah(double x10_inf, double x10_sup)
{
    // Loss is how much the interval [x10_inf, x10_sup] violates the safe bounds.
    double loss_upper = my_relu(x10_sup - 2.8795);
    double loss_lower = my_relu(-2.8795 - x10_inf);
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
    string benchmark_name = "cheetah";

    // --- Variable Declaration ---
    // 17 state variables + 6 action variables = 23 total
    unsigned int numVars = 23;
    intervalNumPrecision = 200;
    Variables vars;
    vector<int> x_ids;
    for (int i = 1; i <= 17; ++i) {
        x_ids.push_back(vars.declareVar("x" + to_string(i)));
    }
    vector<int> u_ids;
    for (int i = 1; i <= 6; ++i) {
        u_ids.push_back(vars.declareVar("u" + to_string(i)));
    }

    // --- Load Learned Dynamics ---
    ifstream dynamics_file("./results/learned_dynamics/Marvelgymnasium_cheetah-v1/model.txt");
    if (!dynamics_file.is_open()) {
        cerr << "Error: Could not open dynamics file." << endl;
        return 1;
    }
    vector<string> dynamics_str;
    string equation;
    for (int i = 0; i < 17; ++i)
    {
        dynamics_file >> equation;
        dynamics_str.push_back(equation);
    }
    // The 6 action variables have no dynamics of their own.
    for (int i = 0; i < 6; ++i) {
        dynamics_str.push_back("0");
    }
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
    // x1-x17: All state variables initialized in [-0.1, 0.1]
    for(int i = 0; i < 17; ++i) {
        X0.push_back(Interval(-0.1, 0.1));
    }
    // u1-u6: actions start at 0
    for(int i = 0; i < 6; ++i) {
        X0.push_back(Interval(0));
    }

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
        for (int i = 0; i < 17; ++i) {
            tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
        }
        
        // Propagate the state through the controller to get the action
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
        polar_setting.set_num_threads(-1);
        TaylorModelVec<Real> tmv_output;
        nn->get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);

        // Assign the controller outputs to the action variables
        for (int i = 0; i < 6; ++i) {
            initial_set.tmvPre.tms[u_ids[i]] = tmv_output.tms[i];
        }

        // Compute the reachable set for the next step using the learned dynamics
        dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);

        if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
        {
            initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
            result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
            
            // Calculate and accumulate the safety loss based on x10
            total_safe_loss += safe_loss_cheetah(inter_box[9].inf(), inter_box[9].sup());

            if (print_safe_sets)
            {
                for(size_t i = 0; i < 17; ++i) {
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
        // Plot x-velocity (x9) vs height (x1)
        plot_setting.setOutputDims("x9", "x1");

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
