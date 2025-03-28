function S=random_points(n,r,q,picture)

% the function generate a collection of n points in R^r in the interval [0,q]^r

S=q*rand(r,n);
if picture==1
    figure
    for i=1:n
        plot3(S(1,i),S(2,i),S(3,i),'b*'); hold on
    end
    grid on
end
end