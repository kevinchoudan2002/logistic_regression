clc,clear;
% setting up the problem
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
[m,n] = size(A);
% hyperparameters
tol = 1e-6;
max_iters = 1000;
x0 = zeros(n,1);

% objective function & gradiant
p = @(w) 1./(1+exp(-b.*(A*w)));
func = @(w) -mean(log(p(w))) + 0.01*w'*w/m;
grad = @(w) -A'*(b.*(1-p(w)))/m + 0.02*w/m;
hess = @(w) A'*(p(w).*(1-p(w)).*A)/m + 0.02*eye(n)/m;

%% Part (a)
% damped newton's method
[func_val, grad_norm] = newton_damped(func, grad, hess, x0, tol, ...
    max_iters);
[func_val_star, ~] = newton_damped(func, grad, hess, x0, 1e-10, 2000);


% newton-CG method
CG_tol1 = @(ng) min(0.5,ng)*ng; 
[func_val1, grad_norm1] = newton_cg(func, grad, hess, x0, tol, ...
    CG_tol1, max_iters); 

CG_tol2 = @(ng) min(0.5,sqrt(ng))*ng; 
[func_val2, grad_norm2] = newton_cg(func, grad, hess, x0, tol, ...
    CG_tol2, max_iters); 

CG_tol3 = @(ng) 0.5*ng; 
[func_val3, grad_norm3] = newton_cg(func, grad, hess, x0, tol, ...
    CG_tol3, max_iters); 


%% Part(b)

% the line search gradient descent 
func_val4 = zeros(max_iters,1);
grad_norm4 = zeros(max_iters,1);
alpha_init = 0.01;
beta = 0.5;
rho = 0.5;
x = x0;
for i = 1:max_iters
    grad_val = grad(x);
    f_init = func(x);

    % backtrack line loop
    alpha = alpha_init;
    while true
        x_new = x - alpha * grad_val;
        f_new = func(x_new);
        % Check Armijo Rules
        if f_new <= f_init - rho * alpha * norm(grad_val)^2
            break;
        end

        alpha = beta * alpha;
    end


    if norm(x_new - x) < tol
        break;
    end
    x = x_new;
    func_val4(i) = func(x);
    grad_norm4(i) = norm(grad(x));
end


%% plot
subplot(1,2,1)

plot(1:i, grad_norm(1:i), 'r', 'LineWidth', 2);
hold on;
plot(1:i, grad_norm1(1:i), 'g', 'LineWidth', 2);
plot(1:i, grad_norm2(1:i), 'b', 'LineWidth', 2);
plot(1:i, grad_norm3(1:i), 'y', 'LineWidth', 2);
plot(1:i, grad_norm4(1:i), 'k', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Gradient Norm');
legend('Newton Method', 'Inexact Newton with CGRule1', ...
    'Inexact Newton with CGRule2','Inexact Newton with CGRule3', ...
    'Backtracking Line Search');

subplot(1,2,2)
plot(1:i, func_val(1:i) - func_val_star(1:i), 'r', 'LineWidth', 2);
hold on;
plot(1:i, func_val1(1:i) - func_val_star(1:i), 'g', 'LineWidth', 2);
plot(1:i, func_val2(1:i) - func_val_star(1:i), 'b', 'LineWidth', 2);
plot(1:i, func_val3(1:i) - func_val_star(1:i), 'y', 'LineWidth', 2);
plot(1:i, func_val4(1:i) - func_val_star(1:i), 'k', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Function Value');
legend('Newton Method', 'Inexact Newton with CGRule1', ...
    'Inexact Newton with CGRule2','Inexact Newton with CGRule3', ...
    'Backtracking Line Search');
