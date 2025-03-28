function [U,S,V] = tsvd(X,r)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [U,S,V] = tsvd(X,r)
%
% TRUNCATED SVD:
%
% CALLS svds(X,r) FOR LARGE MATRICES AND LOW RANK APPROXIMATIONS
%
% OTHERWISE USES eig(X*X') or eig(X'*X), 
% WHICH APPEARS TO BE FASTER
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CUTOFF
cutoff = 4097;

% LARGE MATRICES
m = min(size(X));
assert(r<=m);
if (m>cutoff && r<cutoff)
  [U,S,V] = svds(X,r);
  return;
end

% SMALL MATRICES
X = full(X);
if (size(X,1)<=size(X,2))
  [Q,D] = eig(X*X');
  [maxD,idxD] = maxk(diag(D),r);
  S = diag(sqrt(maxD));
  U = Q(:,idxD);
  V = (X'*U)/S;
else
  [Q,D] = eig(X'*X);
  [maxD,idxD] = maxk(diag(D),r);
  S = diag(sqrt(maxD));
  V = Q(:,idxD);
  U = (X*V)/S;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
