#Lending_Club-Loan_Default_Classification_and_Customer_Segment_Clustering

Peer to peer lending, also referred to as P2P lending is the practice of lending to individuals and businesses through an online platform that matches lenders with borrowers. P2P lending enables individuals
to obtain loans while avoiding financial institutions as the intermediaries. One of the leading P2P lending
platforms is Lending Club, which has now lent over $ 45 billion to more than 3 million customers. The way
the platform operates is that borrowers can create loan listings on the website by filling in an application
that would ask them for details regarding themselves and the loan requirement, such as reason of the loan,
their annual income, credit history etc. Lending Club lists the loan requests within 24 hours for borrowers
who apply and meet their credit policy so that the interested investors can start committing to the invest-
ment. Lending Club follows a verification process on every borrower and once all the required information
is completed and verified, loans are marked as approved. During this verification process, investors can
fund portions of the loans and once the loans are approved, the borrower is issued the loan, provided that
there was sufficient investor commitment. Loan amounts range from $10,000 to $40,000 and investors
can invest as little as $25 per loan. The company makes money through origination and service fees from
borrowers and investors, respectively. Borrowers pay a one-time origination fee of 1.11% to 5% of the
total loan amount while the investors pay a service fee of 1% of each payment received from a borrower [1].
The major reason that the P2P industry is successful is due to its ability to give instant lending
to people in need. However, it does comes up with a major risk, where a borrower might not be able to
repay and the investors might lose their money. 

The major goal of the project is to predict if a borrower 
will be able to repay the loan on the basis of the details provided in their application. This is done by
utilizing the historical loan data from Lending Club to generate a classification model that would predict
whether a loan would be fully paid back or charged (loan is deemed unlikely to be paid back according
to the original terms). This model can be leveraged by Lending Club to make data-driven decisions on
whether a borrower should be lent money by investors or not. To achieve this goal, different classification
models were built by using different supervised machine learning algorithms, such as Logistic Regression,
Random Forest, and Gradient Boosting Machine (GBM) and the model with the highest accuracy is recommended for making decisions. Additionally, an unsupervised machine algorithm, K-means clustering is
also performed to determine different segments of borrowers to gather insights and help the organization
understand its user base (borrowers) better to expand the business. The report outlines the rationale
and methodology that was adopted to build a classification model and generate different clusters, and
also the insights that were gained from the results.
