function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% input_layer_size
% hidden_layer_size
% num_labels
% nn_params_size = size(nn_params)
% Theta1_size = size(Theta1)
% Theta2_size = size(Theta2)
% Theta2_X = size(X)
% Theta2_y = size(y)
% m
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a_1 = X;
a_1 = [ones(m, 1) X];

z_2 = a_1 * Theta1';

a_2 = sigmoid(z_2);

size_a_2 = size(a_2, 1);
a_2 = [ones(size_a_2, 1) a_2];
z_3 = a_2 * Theta2';

a_3 = sigmoid(z_3);

h = a_3;

new_y = zeros(m, num_labels);

for i = 1:m
	new_y(i,y(i)) = 1;
end

J = (1 / m) * sum(sum(-new_y .* log(h) - (1 .- new_y) .* log(1 - h)));

initiat_Theta1 = Theta1(:,2:input_layer_size+1);
initiat_Theta2 = Theta2(:,2:hidden_layer_size+1);

initiat_Theta1_sqr = initiat_Theta1 .^ 2;
initiat_Theta2_sqr = initiat_Theta2 .^ 2;

regularization = (lambda / (2 * m)) * (sum(initiat_Theta1_sqr(:)) + sum(initiat_Theta2_sqr(:)));

J = J + regularization;

% % -------------------------------------------------------------

% % =========================================================================

% % Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];


% end
