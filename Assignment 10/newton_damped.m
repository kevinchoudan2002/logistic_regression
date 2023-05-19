% damped newton method with armijo line search



%imput: func, grad, hess, x0, tol, max_iters
%output: func_vals, grad_vals


function [func_vals, grad_vals] = newton_damped(func, grad_func, hess_func, ...
    x0, tol, max_iters)
    x = x0;
    alpha = 0.01; % Armijo
    beta = 0.5; % Armijo

    % Initialize function value array
    func_vals = zeros(max_iters,1);
    grad_vals = zeros(max_iters,1);
    for i = 1:max_iters
        grad = grad_func(x);
        hess = hess_func(x);
        d = -hess \ grad;

        % Armijo
        t = 1;
        while func(x+t*d) > func(x) + alpha*t*d'*grad
        t = t * beta;
        end

        % Update parameters
        x_new = x + t * d;

        % Check convergence
        if norm(x_new - x) < tol
            break;
        end

        % Update x and function value array
        x = x_new;
        func_vals(i) = func(x);
        grad_vals(i) = norm(grad_func(x));
    end
