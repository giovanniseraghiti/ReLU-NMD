# ReLU_NMD
 markdown
 ReLU-NMD is a relatively new nonlinear matrix decomposition (NMD) model, where, given a sparse non-negative matrix $X$, the goal is to find a low-rank approximation $\Theta$ such that $\max(0,\Theta) \approx X$. ReLU-NMD finds application in entry-dependent matrix completion [^1], compression of sparse data[^2], and manifold learning[^3]. This repository contains state-of-the-art algorithms for solving ReLU-NMD. Specifically, the novel extrapolated block coordinate descent (eBCD-NMD), which is a provably convergent and accelerated method to tackle ReLU-NMD. 
 ReLU-NMD is a relatively new nonlinear matrix decomposition (NMD) model, where, given a sparse non-negative matrix $X$, the goal is to find a low-rank approximation $\Theta$ such that $\max(0,\Theta) \approx X$. ReLU-NMD finds application in entry-dependent matrix completion[1], compression of sparse data[2], and manifold learning[3]. This repository contains state-of-the-art algorithms for solving ReLU-NMD. Specifically, the novel extrapolated block coordinate descent (eBCD-NMD), which is a provably convergent and accelerated method to tackle ReLU-NMD. 
 
 [^1]H. Liu, P. Wang, L. Huang, Q. Qu, and L. Balzano, “Symmetric matrix completion with relu sampling,” arXiv
 [1]H. Liu, P. Wang, L. Huang, Q. Qu, and L. Balzano, “Symmetric matrix completion with relu sampling,” arXiv
 preprint arXiv:2406.05822, 2024.
 
 [^2]G. Seraghiti, A. Awari, A. Vandaele, M. Porcelli, and N. Gillis, “Accelerated algorithms for nonlinear matrix decom-
 [2]G. Seraghiti, A. Awari, A. Vandaele, M. Porcelli, and N. Gillis, “Accelerated algorithms for nonlinear matrix decom-
 position with the relu function,” in 2023 IEEE 33rd International Workshop on Machine Learning for Signal
 Processing (MLSP), pp. 1–6, IEEE, 2023.
 
 [^3]L. K. Saul, “A geometrical connection between sparse and low-rank matrices and its application to manifold learning,”
 [3]L. K. Saul, “A geometrical connection between sparse and low-rank matrices and its application to manifold learning,”
 Transactions on Machine Learning Research, 2022.
