function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
options = [0.01 0.03 0.1 0.3 1 3 10 30];
len = length(options);
choices = zeros(len^2, 2);
for i=1:len
    for j=1:len
        choices(j + (i-1) * len,:) = [options(i) options(j)];
    end;
end;
errors = zeros(len^2, 1);

for i=1:len^2
    model = svmTrain(X, y, choices(i,:)(1), @(x1, x2)gaussianKernel(x1, x2, choices(i,:)(2)));
    predictions = svmPredict(model, Xval);
    errors(i) = mean(double(predictions ~= yval));
    fprintf('error for %d th, %f\n', i, errors(i));
end;
[t, index] = min(errors);
C = choices(index,:)(1);
sigma = choices(index,:)(2);

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







% =========================================================================

end
