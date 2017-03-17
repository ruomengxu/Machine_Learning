function [pro_w_given_z,pro_d_given_z,PI]=mycluster2(bow,K)
% Input:
%     bow = data set. Bag of words representation of text document
%     K = the number of desired topics/clusters. 
%
% Output:
%     pro_w_given_z = p(w|z) 
%     pro_d_given_z = p(d|z)
%     Pi = \pi
%Initialize
doc_size=size(bow,1);
word_size=size(bow,2);
temp=rand(word_size,K);
pro_w_given_z=temp./repmat(sum(temp,1),word_size,1);
temp=rand(doc_size,K);
pro_d_given_z=temp./repmat(sum(temp,1),doc_size,1);
PI=ones(1,K)/K;
pro_z_given_d_w=ones(K,doc_size,word_size);
maxiter=14;
iter=1;
while iter<maxiter
    %Expectation
    fprintf('%d',iter);
    for z=1:1:K
        pro_z_given_d_w(z,:,:)=PI(z)*pro_d_given_z(:,z)*pro_w_given_z(:,z)';
    end
    pro_z_given_d_w=pro_z_given_d_w./repmat(sum(pro_z_given_d_w,1),K,1,1);
    %Maximization
    for z=1:1:K
        pro_w_given_z(:,z)=sum(bow.*squeeze(pro_z_given_d_w(z,:,:)),1)';
        pro_d_given_z(:,z)=sum(bow.*squeeze(pro_z_given_d_w(z,:,:)),2);
        PI(z)=sum(sum(bow.*squeeze(pro_z_given_d_w(z,:,:)),1),2);
    end
    pro_w_given_z=pro_w_given_z./repmat(sum(pro_w_given_z,1),word_size,1);
    pro_d_given_z=pro_d_given_z./repmat(sum(pro_d_given_z,1),doc_size,1);
    PI=PI./sum(PI);
    %
    iter=iter+1;
end
save output pro_w_given_z pro_d_given_z PI;
