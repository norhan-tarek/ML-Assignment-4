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

 % h=X;
 % start=1;
 %stop=0;
 %numLayers=3;
 %layer_size=[25,10];
%for k=1:(numLayers-1)
 %h=[ones(size(h,1),1) h ];
  %  [r,c]=size(h)
   % stop=stop+(c*layer_size(1,k));
    %Theta=reshape(nn_params(start:stop),layer_size(1,k),c);
    %start=c*layer_size(1,k)+ start;
    %size(h)
    %size(Theta)
    %h=sigmoid(h*Theta');
%end

% forwardfeed

X=[ones(m,1) X];

n1= X*Theta1';
n2= sigmoid(n1);
n3=[ones(size(n2,1),1) n2];
h=sigmoid(n3*Theta2');


I=eye(num_labels);
Y=I(y,:);
thetaSqr1=Theta1(:,2:end).^2;
thetaSqr2=Theta2(:,2:end).^2;
J =sum(Y.*log(h)+(1-Y).*log(1-h));
J = ((-1/m)*sum(J));
sumTheta1=sum(sum(thetaSqr1));
sumTheta2=sum(sum(thetaSqr2));
J=J+((lambda/(2*m))*(sumTheta1+sumTheta2));
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
    for t=1:m
        a1= X(t,:);
        z2= sigmoid(a1*Theta1');
        a2=[1 z2];
        z3= sigmoid(a2*Theta2');
        
        d3= z3-Y(t,:);
        g=[1 (a1*Theta1')];
        d2= (d3*Theta2).*sigmoidGradient(g);
        
        d2=d2(2:end);
        
       Theta2_grad = (Theta2_grad + (d3' * a2));
       Theta1_grad = (Theta1_grad + (d2' * a1));
    end
     %unregularized
     %Theta2_grad=Theta2_grad./m;
     %Theta1_grad=Theta1_grad./m;
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%j=0
Theta2_grad(:,1)=(Theta2_grad(:,1))./m;
Theta1_grad(:,1)=(Theta1_grad(:,1))./m;
%j>=1
[i,j]=size(Theta1);
[i1,j2]=size(Theta2);

Theta2_grad(:,2:end)=((Theta2_grad(:,2:end))./m)+ ((lambda/m)*Theta2(:,2:end));
Theta1_grad(:,2:end)=((Theta1_grad(:,2:end))./m)+ ((lambda/m)*Theta1(:,2:end));















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
