function J = solve_SVT(CC,beta)
[U,sigma,VV] = svd(CC,'econ');
sigma = diag(sigma);
svp = length(find(sigma>beta));
if svp>=1
    sigma = sigma(1:svp)-beta;
else
    svp = 1;
    sigma = 0;
end
J = U(:,1:svp)*diag(sigma)*VV(:,1:svp)';
end