function [mean_angular_deviation,jaccard_index] = compare_XY(X,Y,tau)

% computes mean angular deviation and Jaccard index which are error measure for lower-dimensional embeddings

% NORMALIZE
X = bsxfun(@times,X,1./vecnorm(X));
Y = bsxfun(@times,Y,1./vecnorm(Y));

% SETUP
nnzSx = 0;
nnzSy = 0;
nnzI = 0;
an = 0;
ad = 0;
block = 2000;

% LOOP
n = size(Y,2);
for row=1:block:n
  idx = row:min(row+block-1,n);
  XtX = X'*X(:,idx);
  YtY = Y'*Y(:,idx);
  Sx = max(0,XtX-tau);
  Sy = max(0,YtY-tau);
  nnzSx = nnzSx + nnz(Sx);
  nnzSy = nnzSy + nnz(Sy);
  nnzI = nnzI + nnz(Sx&Sy);
  near = find(Sx);
  thetaX = acosd(min(1,XtX(near)));
  thetaY = acosd(min(1,YtY(near)));
  an = an + sum(abs(thetaY-thetaX));
  ad = ad + length(near);
end

nnzU = nnzSx + nnzSy - nnzI;
jaccard_index = nnzI/nnzU;
mean_angular_deviation = an/ad;
return;

