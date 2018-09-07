function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_temp = 0.01;
sigma_temp = 0.01;
err_temp = 0;
num_values = 8;
value_matrix = zeros(num_values*num_values, 3);
i = 1;

while i < num_values+1
	for j = 1:num_values
		model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
		pred = svmPredict(model, Xval);
		err_value = mean(double(pred ~= yval));
		value_matrix(num_values*(i-1)+j, 1) = C_temp;
		value_matrix(num_values*(i-1)+j, 2) = sigma_temp;
		value_matrix(num_values*(i-1)+j, 3) = err_value;
		sigma_temp *= 3;
		j += 1;
	endfor
	C_temp *= 3;
	i += 1;
endwhile

[w, ix] = min(value_matrix(:,3));

C = value_matrix(1, ix);
sigma = value_matrix(2, ix);



% =========================================================================

end
