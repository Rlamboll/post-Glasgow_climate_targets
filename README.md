This set of files allows one to assess the temperatures of scenarios in the paper "A multi-model analysis of post-Glasgow climate targets and feasibility challenges". 


# How to run: 
- Download global emissions from the AR6 database from IIASA and place it in 
	input/AR6_Scenarios_Database_World_v1.0_emissions.csv. This can be done via
	https://data.ene.iiasa.ac.at/ar6/#/workspaces
- run the notebooks beginning 01 and 02. 
	These generate emissions series that can be run through FaIR. 
	They are configured to run with different versions of the 
	There are options in 02 to determine the quantile at which infilling is done, a robustness check explored in the paper.
- Then run the scripts 03 and 04. 
	This will generate the temperature responses required, and a quantiled summary of them. 
- Repeat this process for all desired infilling quantiles. 
- Finally run the notebook 05. 
	This will generate several of the plots used in the paper. 
