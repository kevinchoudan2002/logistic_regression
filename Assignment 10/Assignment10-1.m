clc,clear;
% initialize parameters
rng(mod(21307140016,2^32));
m = 500;
n = 1000;
A = randn(m,n);
b = sign(rand(m,1)-0.5);
x0 = zeros(n,1);



% Set hyperparameters
max_iterations = 3000;
tolerance = 1e-4;

% target function and gradient
p = @(w) 1./(1+exp(-b.*(A*w)));
func = @(w) -mean(log(p(w))) + 0.01*w'*w/m;
grad_func = @(w) -A'*(b.*(1-p(w)))/m + 0.02*w/m;

% function value arrays
func_vals_a1 = zeros(max_iterations,1);
func_vals_a2 = zeros(max_iterations,1);
func_vals_backtrack = zeros(max_iterations,1);


% line-search gradient descent
alpha_init = 0.01;
beta = 0.5;
rho = 0.5;
x = x0;
for i = 1:max_iterations
    grad = grad_func(x);
    f_init = func(x);

    % backtrack line loop
    alpha = alpha_init;
    while true
        x_new = x - alpha * grad;
        f_new = func(x_new);
        % Check Armijo Rules
        if f_new <= f_init - rho * alpha * norm(grad)^2
            break;
        end

        alpha = beta * alpha;
    end


    if norm(x_new - x) < tolerance
        break;
    end
    x = x_new;
    func_vals_backtrack(i) = func(x);
end

% constant step size gradient descent with alpha = 0.1
alpha = 0.1;
x = x0;
for i = 1:max_iterations
    grad = grad_func(x);
    x_new = x - alpha * grad;

    if norm(x_new - x) < tolerance
        break;
    end

    x = x_new;
    func_vals_a1(i) = func(x);
end

% constant step size gradient descent with alpha = 1
alpha = 1;
x = x0;
for i = 1:max_iterations
    grad = grad_func(x);
    x_new = x - alpha * grad;

    if norm(x_new - x) < tolerance
        break;
    end

    x = x_new;
    func_vals_a2(i) = func(x);
end

% Plot function value against iteration number for each method
plot(1:i, func_vals_a1(1:i), 'r', 'LineWidth', 2);
hold on;
plot(1:i, func_vals_a2(1:i), 'g', 'LineWidth', 2);
plot(1:i, func_vals_backtrack(1:i), 'b', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Function Value');
legend('Fixed Step Size (alpha=0.1)', 'Fixed Step Size (alpha=1)', ...
    'Backtracking Line Search');