import random
import pickle
import os
import numpy as np
import pandas as pd


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.project_part = params['project_part'] #useful to be able to use same competition code for each project part
        self.n_items = params["n_items"]
        ########################################################
        self.alpha = 1.0
        self.action_list = []
        
        # self.round_cnt = 0

        ## [DONE]: do we need 2 alpha for part2? or we can use only 1 alpha for 2 items as a whole?
        ## [Decision] use only 1 alpha for all 2 items
        # self.alpha_item0 = 1.0 # for part2.
        # self.alpha_item1 = 1.0 # for part2.

        self.opponent_last_alpha = 1
        self.opponent_alpha_list = [] # record all opponent alpha
        self.opponent_alpha_ratio_large = []
        self.opponent_alpha_ratio_small = []

        self.last_customer_valuation = 1.111111 ## to record the last customer valuation
        self.outlier_detection = 1 ## TODO: for part 1,this should be set to 1, for part 2 this should be set to 1.2

        self.current_customer_valuation_item0 = 0 ## for part 1, this is the only item, for part 2, this is the item0
        self.current_customer_valuation_item1 = 0 ## only for part 2.

        self.price_returned = False
        self.price_stored_for_current_round = [] ## for part 2.

        self.punish_reacted = [] # 0 - not reacted, 1 - reacted (return valuation)

        ########################################################

        # Potentially useful for Part 2 -- 
        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.filename = 'agents/AWinner/trained_model'
        self.trained_model = pickle.load(open(self.filename, 'rb'))

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        # pass

        ############################################################
        # opponent_alpha_ratio = opponent_last_prices / self.last_customer_valuation
        self.process_opponent_alpha(opponent_last_prices)
        ## TODO: remove the first opponent_alpha_ratio since there's no ratio. [DONE]





        ############################################################

    
    ############################# My Tool Functions ###############################

    ### Maintain the opponent alpha ratios
    def process_opponent_alpha(self, opponent_last_prices):
        
        ### First round detection ###
        if self.last_customer_valuation == 1.111111: # do nothing if it's the first round
            return
        
        self.opponent_last_alpha = opponent_last_prices / self.last_customer_valuation
        self.opponent_alpha_list.append(self.opponent_last_alpha)

        ### Outlier Detection ###
        if opponent_last_prices > self.last_customer_valuation * self.outlier_detection:
            pass # do nothing
        else:
            ## TODO: determine whether the index is correct.
            # last_opponent_alpha_ratio = self.opponent_last_alpha / self.opponent_alpha_list[len(self.opponent_alpha_list) - 2]
            last_opponent_alpha_ratio = self.opponent_last_alpha / self.opponent_alpha_list[-2]
            if last_opponent_alpha_ratio >= 1:
                self.opponent_alpha_ratio_large.append(last_opponent_alpha_ratio)
            else:
                self.opponent_alpha_ratio_small.append(last_opponent_alpha_ratio)

    ### Determine the alpha mechanisms of our opponent
    def is_only_adaptive_to_themself(self):
        # for the small ratios:
        mean_value_small = np.mean(self.opponent_alpha_ratio_small)
        rms_diff_small = np.sqrt(np.mean((self.opponent_alpha_ratio_small - mean_value_small) ** 2))

        # for the large ratios:
        mean_value_large = np.mean(self.opponent_alpha_ratio_large)
        rms_diff_large = np.sqrt(np.mean((self.opponent_alpha_ratio_large - mean_value_large) ** 2))

        ## TODO: finalize the decision boundaries.
        ### could use correlation functions of numpy.
        if rms_diff_small < 0.1 & rms_diff_large < 0.1:
            return True
        else:
            return False

    ### Predict opponent's alpha
    def predict_opponent_alpha_ratio(self, tendance):
        if self.is_only_adaptive_to_themself() == True:
            if tendance == 'decrease':
                return np.mean(self.opponent_alpha_ratio_small)
            else:
                return np.mean(self.opponent_alpha_ratio_large)
        else: ## is adaptive to themselves and our alpha
            ## TODO
            pass

    ## if opponent price is zero, call this function to react to their punishments
    # def zero_punishment_reaction(self, opponent_last_prices):
    #     if self.project_part == 1:
    #         ## TODO: add punishments for part1.
    #         pass

    #     if self.project_part == 2:
    #         # if last opponent price is zero, return customer valuation
    #         ## TODO: check this logic for 2 items
    #         if opponent_last_prices[0] == 0 or opponent_last_prices[1] == 0:
    #             min_value = 0.7
    #             max_value = 0.8
    #             scaled_random_number = min_value + (max_value - min_value) * np.random.rand()
    #             self.price_returned = True
    #             self.price_stored_for_current_round = [self.current_customer_valuation_item0 * scaled_random_number, self.current_customer_valuation_item1 * scaled_random_number]

    def return_random_alpha(self):
        min_value = 0.7
        max_value = 0.99
        return min_value + (max_value - min_value) * np.random.rand()
    
    def zero_punishment_reaction_level_1(self):
        ## TODO: threshold settings [DONE]
        if self.opponent_last_alpha == 0 or (self.opponent_alpha_list[-2] - self.opponent_last_alpha) / self.opponent_alpha_list[-2] > 0.5:
            min_value = 0.8
            max_value = 0.9
            scaled_random_number = min_value + (max_value - min_value) * np.random.rand()
            self.price_returned = True
            self.price_stored_for_current_round = [self.current_customer_valuation_item0 * scaled_random_number, self.current_customer_valuation_item1 * scaled_random_number]
            # self.punish_reacted.append(1)
        # else:
            # self.punish_reacted.append(0)

    def zero_punishment_reaction_level_2(self):
        if self.opponent_last_alpha == 0 or (self.opponent_alpha_list[-2] - self.opponent_last_alpha) / self.opponent_alpha_list[-2] > 0.5:
            min_value = 0.3
            max_value = 0.5
            scaled_random_number = min_value + (max_value - min_value) * np.random.rand()
            self.price_returned = True
            self.price_stored_for_current_round = [self.current_customer_valuation_item0 * scaled_random_number, self.current_customer_valuation_item1 * scaled_random_number]


    def punish_opponent(self):
        if self.action_list[-2] == 1:
            self.price_returned = True
            random_price = (400 + (450 - 400) * np.random.rand())
            self.price_stored_for_current_round = [random_price, random_price]
        elif self.opponent_last_alpha < (0.3 + (0.4 - 0.3) * np.random.rand()):
            self.alpha = (0 + (0.15 - 0) * np.random.rand())
            self.price_returned = True
            self.price_stored_for_current_round = [self.current_customer_valuation_item0 * self.alpha, self.current_customer_valuation_item1 * self.alpha]
            self.action_list[-1] = 1 # punished

    ############################# END: My Tool Functions ###############################


    ### Generating optimal prices ###
    def generate_optimal_prices(self, new_buyer_covariates):
        price_sequence = np.linspace(0, 100, 10)
        price_sequence_0 = np.linspace(50, 1000, 10)

        # user_single = user.to_frame().T
        user_single = pd.Series(new_buyer_covariates)
        user_single.index = ['Covariate1', 'Covariate2', 'Covariate3']
        user_single = user_single.to_frame().T
        user = pd.concat([user_single]*100, ignore_index=True)

        for i in range(len(user)):
            user.at[i, 'price_item_0'] = price_sequence_0[(i // len(price_sequence_0)) - 1]
            user.at[i, 'price_item_1'] = price_sequence[i % len(price_sequence)]

        # self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0]

        probabilities = self.trained_model.predict_proba(user)
        # probabilities = grid_search.predict_proba(user)

        user[['p0', 'p1', 'p2']] = probabilities
        user['revenue'] = user['price_item_0'] * user['p0'] + user['price_item_1'] * user['p1']

        ## finding 
        optimal_price_0 = user.loc[np.argmax(user['revenue']), 'price_item_0']
        optimal_price_1 = user.loc[np.argmax(user['revenue']), 'price_item_1']

        print(" ========== optimal_price_0", optimal_price_0)
        print(" ========== optimal_price_1", optimal_price_1)

        ## narrow down the range and find a better price
        tmp = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        price_sequence_optimized_item_0 =  np.array([optimal_price_0]*11) + np.array(tmp)
        price_sequence_optimized_item_1 =  np.array([optimal_price_1]*11) + np.array(tmp)

        user_optimized = pd.concat([user_single]*100, ignore_index=True)
        for i in range(len(user)):
            user_optimized.at[i, 'price_item_0'] = price_sequence_optimized_item_0[(i // len(price_sequence_0)) - 1]
            user_optimized.at[i, 'price_item_1'] = price_sequence_optimized_item_1[i % len(price_sequence)]

        probabilities = self.trained_model.predict_proba(user_optimized)
        # probabilities = grid_search.predict_proba(user_optimized)

        user_optimized[['p0', 'p1', 'p2']] = probabilities
        user_optimized['revenue'] = user_optimized['price_item_0'] * user_optimized['p0'] + user_optimized['price_item_1'] * user_optimized['p1']
        # user_optimized['r1'] = user_optimized['price_item_1'] * user_optimized['p1']
        # user_optimized['r0_null'] = user_optimized['price_item_0'] * user_optimized['p2']
        # user_optimized['r1_null'] = user_optimized['price_item_1'] * user_optimized['p2']

        # optimal_optimal price
        optimal_optimal_price_0 = user_optimized.loc[np.argmax(user_optimized['revenue']), 'price_item_0']
        optimal_optimal_price_1 = user_optimized.loc[np.argmax(user_optimized['revenue']), 'price_item_1']

        ## expected_revenue = user_optimized.loc[np.argmax(user_optimized['revenue']), 'revenue']

        print(" ========== optimal_optimal_price_0", optimal_optimal_price_0)
        print(" ========== optimal_optimal_price_1", optimal_optimal_price_1)

        # finding the probability with maximal revenue
        p0 = user_optimized.loc[np.argmax(user_optimized['revenue']), 'p0']
        p1 = user_optimized.loc[np.argmax(user_optimized['revenue']), 'p1']
            
        print(" +++++++++++ p0: ", p0)
        print(" +++++++++++ p1: ", p1)

        revenue_0 = optimal_optimal_price_0 * p0
        revenue_1 = optimal_optimal_price_1 * p1

        print(" +++++++++++ revenue_0: ", revenue_0)
        print(" +++++++++++ revenue_1: ", revenue_1)

        # expected_revenue
        # expected_revenue = revenue_0 + revenue_1
        # expected_revenue = optimal_optimal_price_0 * p0 + optimal_optimal_price_1 * p1 

        return [optimal_optimal_price_0, optimal_optimal_price_1]

    
        



    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items, indicating prices this agent is posting for each item.
    def action(self, obs):

        # For Part 1, new_buyer_covariates will simply be a vector of length 1, containing a single numeric float indicating the valuation the user has for the (single) item
        # For Part 2, new_buyer_covariates will be a vector of length 3 that can be used to estimate demand from that user for each of the two items
        new_buyer_covariates, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)

        #################################################
        ######### Maintaining self parameters ##############
        # self.round_cnt = self.round_cnt + 1
        self.action_list.append(0)


        ######### END: Maintaining self parameters #########

        if self.project_part == 1:
            self.current_customer_valuation_item0 = new_buyer_covariates[0]
        if self.project_part == 2:
            self.current_customer_valuation_item0 ## TODO: maintain customer valuations from the model.
            self.current_customer_valuation_item1 ## TODO: maintain customer valuations from the model.


        # self.zero_punishment(last_sale[2][self.opponent_number])

        #################################################

        # Potentially useful for Part 1 --
        # Currently output is just a deterministic price for the item, but students are expected to use the valuation (inside new_buyer_covariates) and history of prices from each team to set a better price for the item
        if self.project_part == 1:
            return [3]

        # Potentially useful for Part 2 -- 
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to set prices for each item.
        if self.project_part == 2:
            ########## setting alpha ##########
            ## First Stage: return ramdon alpha for ?? rounds
            if self.this_agent_number < 30 :
                self.alpha = self.return_random_alpha()

            ## Second Stage: return prices based on predicted opponent alpha
            if self.this_agent_number >= 30:
                self.zero_punishment_reaction_level_1()
                self.punish_opponent()

                if self.price_returned:
                    # TODO: directly return the stored price
                    pass
                else:
                    did_customer_buy_from_me = last_sale[1] == self.this_agent_number


                    if did_customer_buy_from_me:  # can increase prices
                        self.alpha = np.mean(self.opponent_alpha_ratio_small) - (0.1 + (0.12 - 0.1) * np.random.rand())
                    else:  # should decrease prices
                        self.alpha = np.mean(self.opponent_alpha_ratio_small) - (0.13 + (0.16 - 0.13) * np.random.rand())
                ## TODO: maintain accumulative revenue amount to choose from level 1 and level 2.
            
            ########## END: setting alpha ##########

            optimal_price = np.array(self.generate_optimal_prices(new_buyer_covariates)) * (0.7 + (0.75 - 0.7) * np.random.rand())

            # return self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0] + random.random()
            return optimal_price
        

        ########################################################################
        # Maintain the last customer valution for the next round

        ## cleaning flag values
        self.price_returned = False


        if self.project_part == 1:
            self.last_customer_valuation = new_buyer_covariates [0]

        if self.project_part == 2:
            ## TODO: maintain self.last_customer_valuation with our model predicted price.
            pass

        ########################################################################