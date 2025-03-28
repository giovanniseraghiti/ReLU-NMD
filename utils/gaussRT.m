function [sumLogP,means,sumVrnc] = gaussRT(mu,sigmaSq)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [sumLogP,means,sumVrnc] = gaussRT(mu,sigmaSq)
% 
% COMPUTES STATISTICS OF A RIGHT TRUNCATED GAUSSIAN DISTRIBUTION 
% WITH SUPPORT ON (-Inf,0]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigma = sqrt(sigmaSq);
gamma = mu/sigma;
ratio = sqrt(2/pi)./erfcx(gamma/sqrt(2));

means = mu - (sigma .* ratio);
sumVrnc = sigmaSq * sum(1-(ratio-gamma).*ratio);

logP = -(0.5*log(2*pi) + log(ratio) + 0.5*gamma.*gamma);
logP(ratio==0) = 0;
sumLogP = sum(logP);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
