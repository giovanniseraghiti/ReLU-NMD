function s=set_threshold(X)

% function used to visualize the 1 Nearest Neighbourhood between columns of the data set X considered. 
% The visual estimate is used to set the parameter tau in the TSM framework. 
% Ideally, we should choose tau such that each columns has at least one other element in the data set,
% such that the cosine between them is greater that tau. 

% Compute the norm of each column
norms = sqrt(sum(X.^2, 1));

% Normalize each column
normalized_X = X./ norms;

% Compute the cosine similarity matrix
S= rad2deg(acos(max(-1,min(1,normalized_X' * normalized_X))));

[s,ind]=min(S+90*eye(size(S,1),size(S,2))); %row vector with maximum element for each column, ind contains row index

s1=[];
for i=1:size(S,2)
    s1=[s1;S(1:ind-1,i);S(ind+1:end,i)];
end

% Plot the first histogram with 'pdf' normalization
histogram(s, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'DisplayName', 'Data 1');
hold on;
yyaxis right;
% Plot the second histogram with 'probability' normalization
histogram(s1, 'Normalization', 'probability', 'FaceAlpha', 0.5, 'DisplayName', 'Data 2');

yyaxis left;
ylabel('PDF'); % Label for the primary y-axis
yyaxis right;
ylabel('Probability'); % Label for the secondary y-axis

% Add labels and title
xlabel('Value');
title('Histograms with Different Normalizations');
legend('show'); % Show legend to differentiate the histograms

end