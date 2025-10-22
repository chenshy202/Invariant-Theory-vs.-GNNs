import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import extract_values

device = 'cuda' if torch.cuda.is_available() else 'cpu'


mpl.rcParams.update({
   
    "font.size": 12,      
    "axes.labelsize": 24,    
    "axes.titlesize": 24,   
    "xtick.labelsize": 19,   
    "ytick.labelsize": 19,   
    "legend.fontsize": 26,   
    "figure.titlesize": 28,
    
   
    'lines.linewidth': 5,       
    'lines.markersize': 12,     
    'errorbar.capsize': 7       
})

if __name__ == "__main__":
    results_train_p = extract_values("results_PointClouds/PointGNN/results_train.txt")
    results_test_p = extract_values("results_PointClouds/PointGNN/results_test.txt")
    results_time_p = extract_values("results_PointClouds/PointGNN/results_time.txt")

    train_means_p = [np.mean(res) for res in results_train_p]
    train_stderr_p = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_train_p]
    test_means_p = [np.mean(res) for res in results_test_p]
    test_stderr_p = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_test_p]
    time_means_p = [np.mean(res) for res in results_time_p]
    time_stderr_p = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_time_p]

    results_train_in = extract_values("results_PointClouds/Invariantf/results_train.txt")
    results_test_in = extract_values("results_PointClouds/Invariantf/results_test.txt")
    results_time_in = extract_values("results_PointClouds/Invariantf/results_time.txt")
    
    train_means_in = np.mean(results_train_in[0])
    train_stderr_in = np.std(results_train_in[0], ddof=1)
    test_means_in = np.mean(results_test_in[0])
    test_stderr_in = np.std(results_test_in[0], ddof=1)
    time_means_in = np.mean(results_time_in[0])
    time_stderr_in = np.std(results_time_in[0], ddof=1)

    train_means_inl = np.full(14, train_means_in)
    train_stderr_inl = np.full(14, train_stderr_in)
    test_means_inl = np.full(14, test_means_in)
    test_stderr_inl = np.full(14, test_stderr_in)
    time_means_inl = np.full(14, time_means_in)
    time_stderr_inl = np.full(14, time_stderr_in)


    results_train_inkm = extract_values("results_PointClouds/Invariantf/kmeans_train.txt")
    results_test_inkm = extract_values("results_PointClouds/Invariantf/kmeans_test.txt")
    results_time_inkm = extract_values("results_PointClouds/Invariantf/kmeans_time.txt")
    
    train_means_inkm = np.mean(results_train_inkm[0])
    train_stderr_inkm = np.std(results_train_inkm[0], ddof=1)
    test_means_inkm = np.mean(results_test_inkm[0])
    test_stderr_inkm = np.std(results_test_inkm[0], ddof=1)
    time_means_inkm = np.mean(results_time_inkm[0])
    time_stderr_inkm = np.std(results_time_inkm[0], ddof=1)

    train_means_inkml = np.full(14, train_means_inkm)
    train_stderr_inkml = np.full(14, train_stderr_inkm)
    test_means_inkml = np.full(14, test_means_inkm)
    test_stderr_inkml = np.full(14, test_stderr_inkm)
    time_means_inkml = np.full(14, time_means_inkm)
    time_stderr_inkml = np.full(14, time_stderr_inkm)



    r_values = [0.06, 0.08] + list(np.arange(0.1, 1.05, 0.1)) + [1.2, 1.4]

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, train_means_inl, linestyle='--', color='red')

    plt.errorbar(0.75, train_means_in, yerr=train_stderr_in, fmt='-s', color='red',
                capsize=5, label="DS-CI")
    plt.plot(r_values, train_means_inkml, linestyle='-.', color='orange')
    plt.errorbar(0.75, train_means_inkm, yerr=train_stderr_inkm, fmt='-^', color='orange',
                capsize=5, label="OI-DS")
    plt.errorbar(r_values, train_means_p, yerr=train_stderr_p, fmt='-o', label='PointGNN', color='#1f77b4', capsize=5)
    # plt.errorbar(r_values, train_means_e, yerr=train_stderr_e, fmt='->', label='EGNN', color='#1f77b4', capsize=5)
    plt.xlabel('r (radius)')
    plt.ylabel('Rank correlation')
    # plt.title('Comparison of Performance (training)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/train.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, test_means_inl, linestyle='--', color='red')
    plt.errorbar(0.75, test_means_in, yerr=test_stderr_in, fmt='-s', color='red',
                capsize=5, label="DS-CI")
    plt.plot(r_values, test_means_inkml, linestyle='-.', color='orange')
    plt.errorbar(0.75, test_means_inkm, yerr=test_stderr_inkm, fmt='-^', color='orange',
                capsize=5, label="OI-DS")
    plt.errorbar(r_values, test_means_p, yerr=test_stderr_p, fmt='-o', label='PointGNN', color='#1f77b4', capsize=5)

    plt.xlabel('r (radius)')
    plt.ylabel('Rank correlation')
    # plt.title('Comparison of Performance (test)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/test.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, time_means_inl, linestyle='--', color='red')

    plt.errorbar(0.75, time_means_in, yerr=time_stderr_in, fmt='-s', color='red',
                capsize=5, label="DS-CI")
    plt.plot(r_values, time_means_inkml, linestyle='-.', color='orange')
    plt.errorbar(0.75, time_means_inkm, yerr=time_stderr_inkm, fmt='-^', color='orange',
                capsize=5, label="OI-DS")
    plt.errorbar(r_values, time_means_p, yerr=time_stderr_p, fmt='-o', label='PointGNN', color='#1f77b4', capsize=5)

    plt.xlabel('r (radius)')
    plt.ylabel('Time')
    # plt.title('Comparison of Running Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/time.png", dpi=300, bbox_inches='tight')
    plt.show()
