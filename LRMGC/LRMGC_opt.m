%% function LRMGC_opt
function [Z,C, error] = LRMGC_opt(X, lambda1, lambda2, p, maxIter)
%% normalize data to lie in [-1 1]
X = X./(repmat(sqrt(sum(X.^2,1)),size(X,1),1)+eps);
[~, N] = size(X);

%% Initialization S by KNN
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Cosine';
S = constructW(X',options);
D = diag(sum(S'));
LL = D - S;

%% ADMM parameters
rho = 2.8;
miu = 0.01;
miu_max = 1e8;

%% Initialization
C = zeros(N,N);
A = zeros(N,N);
Z = zeros(N, N);
J = zeros(N, N);
L = zeros(N, N);
R = zeros(N, N);
E = zeros(size(X));
Q1 = zeros(size(X));
Q2 = zeros(N, N);
Q3 = zeros(N, N);
Q4 = zeros(N, N);

for iter = 1:maxIter
    %% update E
    tempE = X - X*Z + Q1/miu;
    for i=1:N
        nw = norm(tempE(:,i));
        if nw>lambda1/miu
            x = (nw-lambda1/miu)*tempE(:,i)/nw;
        else
            x = zeros(length(tempE(:,i)),1);
        end
        tempE(:,i) = x;
    end
    E = tempE;
    
    %% updata Z
    M1 = X - E + Q1/miu;
    M2 = L*C*R' - Q2/miu;
    M3 = J - Q4/miu;
    Z = (X'*X+2*eye(N))\(X'*M1+M2+M3);
    
    %% updata J
    tempJ = Z + Q4/miu;
    J = (tempJ + tempJ')/2;
    
    %% update L ang R
    tempL = (Z+Q2/miu)*R*C';
    [u,~,vv] = svd(tempL,'econ');
    L = u*vv';
    
    tempR = (Z+Q2/miu)'*L*C;
    [u,~,vv] = svd(tempR,'econ');
    R = u*vv';
    
    %% updata C
    tempC = 0.5*(L'*(Z+Q2/miu)*R+A-Q3/miu);
    [UUU,sigma,VVV] = svd(tempC,'econ');
    sigma = diag(sigma);
    xi = spw(sigma,1/(2*miu),p);
    C = UUU*diag(xi)*VVV';
    
    %% updata A
    tempA = C + Q3/miu;
    A = miu/lambda2*tempA/(LL'+LL+miu/lambda2*eye(N));
    
    %% update Q1,Q2,Q3 and mu
    Q1 = Q1 + miu*(X-X*Z-E);
    Q2 = Q2 + miu*(Z-L*C*R');
    Q3 = Q3 + miu*(C-A);
    Q4 = Q4 + miu*(Z-J);
    miu = min(rho*miu,miu_max);
    
    %% 
    tempstop = Z-L*C*R';
    temp_ter1 = max(max(abs(tempstop)));
    stop = temp_ter1;
    error(iter) = stop;
    %------------------------------------------------------------------------
    disp(['iter' num2str(iter) 'stop' num2str(stop)])
    if abs(stop)<1e-5
        break
    end
end

end