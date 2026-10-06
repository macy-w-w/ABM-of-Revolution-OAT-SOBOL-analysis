import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import os

sns.set_style("white")
sns.set_context("talk")

from SALib.sample import sobol as sobolsample
from SALib.analyze import sobol

# Model parameters
problem = {
    'num_vars': 8,
    'names': [
        'k1', 'k2', 'government_legitimacy', 'max_jail_term',
        'initial_citizen_density', 'initial_policeman_density',
        'initial_revolutionary_density', 'vision'
    ],
    'bounds': [
        [1.5, 3.5], [0.01, 0.05], [0.05, 0.99], [5, 50],
        [40, 80], [1, 10], [1, 10], [0.1, 10.0]
    ]
}

fixed_params = {
    'a': 0.5, 'b': 0.5, 'f': 0.1, 'n': 1.2,
    'policeman_precision': 0.40, 'revolutionary_precision': 0.30
}

def run_single_simulation(experiment):
    """Run a single NetLogo simulation."""
    import pynetlogo
    import sys
    
    try:
        sys.stdout.write(".")  # Print dot to show it's starting
        sys.stdout.flush()
        
        netlogo = pynetlogo.NetLogoLink(gui=False)
        
        sys.stdout.write("L")  # L for Link
        sys.stdout.flush()
        
        netlogo.load_model('./models/Rev_Model_with_parameters.nlogox')
        
        sys.stdout.write("M")  # M for Model
        sys.stdout.flush()
        
        # Name mapping
        name_mapping = {
            'k1': 'k1',
            'k2': 'k2',
            'government_legitimacy': 'government-legitimacy',
            'max_jail_term': 'max-jail-term',
            'initial_citizen_density': 'initial-citizen-density',
            'initial_policeman_density': 'initial-policeman-density',
            'initial_revolutionary_density': 'initial-revolutionary-density',
            'vision': 'vision'
        }
        
        for i, name in enumerate(problem['names']):
            netlogo_name = name_mapping[name]
            if netlogo_name in ['max-jail-term', 'vision']:
                netlogo.command(f"set {netlogo_name} {round(experiment[i])}")
            else:
                netlogo.command(f"set {netlogo_name} {experiment[i]}")
        
        netlogo.command(f"set a {fixed_params['a']}")
        netlogo.command(f"set b {fixed_params['b']}")
        netlogo.command(f"set f {fixed_params['f']}")
        netlogo.command(f"set n {fixed_params['n']}")
        netlogo.command(f"set policeman_precision {fixed_params['policeman_precision']}")
        netlogo.command(f"set revolutionary_precision {fixed_params['revolutionary_precision']}")
        
        sys.stdout.write("S")  # S for Setup
        sys.stdout.flush()
        netlogo.command("setup")
        
        sys.stdout.write("R")  # R for Running
        sys.stdout.flush()
        for _ in range(500):
            netlogo.command("go")
        
        results = {
            'final_active_citizens': float(netlogo.report("count citizens with [active?]")),
            'final_total_citizens': float(netlogo.report("count citizens")),
            'final_policemen': float(netlogo.report("count policemen")),
            'final_revolutionaries': float(netlogo.report("count revolutionaries")),
            'final_jailed_citizens': float(netlogo.report("count citizens with [jail-term > 0]")),
            'revolution_started': float(netlogo.report("t_rev") < 999),
            'revolution_time': float(netlogo.report("t_rev")),
            'mean_grievance': float(netlogo.report("mean [grievance] of citizens")),
            'mean_hardship': float(netlogo.report("mean [hardship] of citizens")),
            'proportion_active': float(netlogo.report("count citizens with [active?]")) / 
                                max(1.0, float(netlogo.report("count citizens")))
        }
        
        netlogo.kill_workspace()
        sys.stdout.write("D")  # D for Done
        sys.stdout.flush()
        return results
        
    except Exception as e:
        sys.stdout.write(f"E({str(e)[:20]})")
        sys.stdout.flush()
        try:
            netlogo.kill_workspace()
        except:
            pass
        return {
            'final_active_citizens': 0, 'final_total_citizens': 0,
            'final_policemen': 0, 'final_revolutionaries': 0,
            'final_jailed_citizens': 0, 'revolution_started': 0,
            'revolution_time': 500, 'mean_grievance': 0,
            'mean_hardship': 0, 'proportion_active': 0
        }

if __name__ == '__main__':
    # Generate parameter samples
    N_base = 32
    param_values = sobolsample.sample(problem, N_base)
    
    print(f"Generated {param_values.shape[0]} parameter combinations")
    print("Parameter ranges being sampled:")
    for i, name in enumerate(problem['names']):
        print(f"  {name}: {problem['bounds'][i]}")
    
    # Run simulations using multiprocessing
    num_cores = min(cpu_count(), 4)  # Use up to 4 cores
    print(f"\nRunning simulations using {num_cores} cores...")
    
        # Run FIRST simulation as a test
    print("Testing single simulation first...")
    test_result = run_single_simulation(param_values[0])
    print(f"\nTest result: {test_result}")
    
    if test_result['final_total_citizens'] == 0:
        print("ERROR: Test simulation failed! Check NetLogo setup.")
        exit()
    
    print("\nTest passed! Starting full batch...")
    
    # Now run all simulations
    num_cores = min(cpu_count(), 4)
    print(f"Running {len(param_values)} simulations using {num_cores} cores...")
    
    with Pool(processes=num_cores) as pool:
        all_results = pool.map(run_single_simulation, param_values)
    
    # Convert to DataFrame
    results = pd.DataFrame(all_results)
    print(f"\nCompleted! Results shape: {results.shape}")
    print("\nFirst 5 rows:")
    print(results.head())
    
    # Prepare outputs for sensitivity analysis
    outputs = {
        'active_citizens': results['final_active_citizens'].values,
        'policemen': results['final_policemen'].values,
        'revolutionaries': results['final_revolutionaries'].values,
        'revolution_time': results['revolution_time'].values,
        'proportion_active': results['proportion_active'].values
    }
    
    # Perform sensitivity analysis
    print("\n=== SENSITIVITY ANALYSIS RESULTS ===")
    for output_name, Y in outputs.items():
        print(f"\n--- {output_name.upper()} ---")
        try:
            Si = sobol.analyze(problem, Y, print_to_console=False)
            
            print(f"{'Parameter':<25} {'S1 (First-order)':<15} {'ST (Total-order)':<15}")
            print("-" * 55)
            for i, name in enumerate(problem['names']):
                print(f"{name:<25} {Si['S1'][i]:<15.4f} {Si['ST'][i]:<15.4f}")
        except Exception as e:
            print(f"Could not analyze {output_name}: {e}")
    
    # Visualization
    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 5*len(outputs)))
    if len(outputs) == 1:
        axes = [axes]
    for ax, (output_name, Y) in zip(axes, outputs.items()):
        try:
            Si = sobol.analyze(problem, Y, print_to_console=False)
            
            x = np.arange(len(problem['names']))
            width = 0.35
            
            ax.bar(x - width/2, Si['S1'], width, label='First-order (S1)')
            ax.bar(x + width/2, Si['ST'], width, label='Total-order (ST)')
            
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Sensitivity Index')
            ax.set_title(f'Sensitivity Analysis: {output_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(problem['names'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Analysis failed: {e}",
                    ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Revolution statistics
    revolution_success = results['revolution_started'].values
    revolution_rate = np.mean(revolution_success)
    print(f"\n=== REVOLUTION STATISTICS ===")
    print(f"Probability of revolution: {revolution_rate:.3f}")
    
    # Save results
    import json
    results_dict = {
        'problem': problem,
        'fixed_params': fixed_params,
        'N': N_base,
        'max_ticks': 500,
        'valid_simulations': len(results),
        'parameter_values': param_values.tolist(),
        'outputs': {k: v.tolist() for k, v in outputs.items()},
        'revolution_success': revolution_success.tolist()
    }
    
    with open('simulation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\nResults saved to 'simulation_results.json'")
    print("Sensitivity analysis plot saved to 'sensitivity_analysis.png'")
