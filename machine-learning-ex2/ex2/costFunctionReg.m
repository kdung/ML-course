function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);
y1 = y .* log(h);
y2 = (1 - y) .* log(1 - h);
n = size(theta,1);
thetaFrom2 = theta(2:n,:);
reg = (sum(thetaFrom2.^2)) * lambda /(2 * m);
J = sum(-y1 - y2)/m + reg;
error = (h - y);
regGrad = thetaFrom2 .* (lambda/m);
grad0 = sum(error .* X(1))/m;
XFrom1 = X(:,2:size(X,2));
gradFrom2 = sum(error .* XFrom1)/m + regGrad';
grad = [grad0 gradFrom2];



% =============================================================

end
