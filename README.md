# ReLU_NMD
 ReLU matrix decomposition is a relatively new nonlinear matrix decomposition (NMD) model, where, given a sparse non-negative matrix $X$, we aim at finding a low-rank approximation $\Theta$ such that $\max(0,\Theta) \approx X$. Specifically, one can compute a rank- $r$ decomposition of $X$ solving the Latent-ReLU-NMD model, that is 
 
    $$ \lVert Z - \Theta \rVert_F^2 \ \mbox{such that} \ \max(0,Z)=X,  \ \mbox{and} \  rank(\Theta) \leq r,$$    

or the Three-block-ReLU-NMD (3B-ReLU-NMD) formulation:

         $$ \lVert Z - WH \rVert_F^2 \ \mbox{such that} \ \max(0,Z)=X.$$ 
         
ReLU decomposition finds application in entry-dependent matrix completion [^1], in the recovery of Euclidean distance matrices, in the compression of sparse data[^2], and in manifold learning[^3]. This repository contains state-of-the-art algorithms for computing ReLU decompositions and examples of test problems.
Description of the algorithms:
 1. Naive-NMD: alternate optimization scheme that computes at each iteration one global optima of the subproblem for $Z$ and $\Theta$ in the Latent-ReLU-NMD formulation[3],
 2. Aggressive-NMD (A_NMD): adaptively extrapolated version of the Naive-NMD scheme[2],
 3. Expectation-minimization NMD (EM-NMD): expectation-minimization framework applied to the Latent-ReLU-NMD problem[3],
 4. Block coordinate descent NMD (BCD-NMD): block coordinate descent algorithm which computes one global optima for each subproblem of the 3B-ReLU-NMD formulation[2],
 5. Extrapolade BCD-NMD (eBCD-NMD): extrapolated and provably convergent variant of the BCD-NMD[].
 6. Three block NMD (3B-NMD): heuristic extrapolation technique applied to the BCD-NMD scheme[2].

The numerical experiments folder contains codes to replicate the experiments contained in the paper :
 1. Figure5_1: Synthetic matrix completion example with ReLU sampling,
 2. Figure5_2: Euclidean distance matrix completion example recovering distance matrix of points with different distribution in a three-dimensional space,
 3. Figure5_3: example using the Threshold Similarity Matching (TSM) approach[3] to compute lower-dimensional embedding of text data,
 4. Figure5_4: script to visually observe compressed image using eBCD-NMD algorithm and compare with the truncated singular value approximation of the same rank,
 5. Table5_1:  uses ReLU decomposition to compress sparse data and images.

Other folders description:
 1. Datasets: contains the data sets needed to run the codes,
 2. utils: contains the functions needed to run the main codes,
 3. Rank_1_modified_codes_for_EDMC: codes that solves a rank-1 modified version of the Latent-ReLU-NMD and 3B-ReLU-NMD needed for the recovery of Euclidean distance matrices, more details in [].
 
 [1]H. Liu, P. Wang, L. Huang, Q. Qu, and L. Balzano, “Symmetric matrix completion with relu sampling,” arXiv
 preprint arXiv:2406.05822, 2024.
 
 [2]G. Seraghiti, A. Awari, A. Vandaele, M. Porcelli, and N. Gillis, “Accelerated algorithms for nonlinear matrix decom-
 position with the relu function,” in 2023 IEEE 33rd International Workshop on Machine Learning for Signal
 Processing (MLSP), pp. 1–6, IEEE, 2023.
 
 [3]L. K. Saul, “A geometrical connection between sparse and low-rank matrices and its application to manifold learning,”
 Transactions on Machine Learning Research, 2022.
