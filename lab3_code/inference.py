import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    # forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    # TODO: Compute the forward messages
    for i in range(0, num_time_steps):
        forward_messages[i] = rover.Distribution()
        if i == 0:
            observe_x0y0 = observations[0]
            for zi in prior_distribution:  # position represent (xn, yn)
                # a(z0) = P(z0) * P((x0, y0)|z0)
                forward_messages[i][zi] = prior_distribution[zi] * observation_model(zi)[observe_x0y0]
            forward_messages[i].renormalize()

        else:
            observe_xiyi = observations[i]
            for zi in all_possible_hidden_states:
                # TODO: add part 2 here
                if observe_xiyi == None:
                    P_obs_cond_zi = 1
                else:
                    # alpha(zi) = P((xi_obs, yi_obs) | zi) * sum(P(zi | zi - 1) * alpha(zi - 1))
                    # P((xi_obs, yi_obs) | zi) is named P_obs_cond_zi
                    P_obs_cond_zi = observation_model(zi)[observe_xiyi]

                # calculate sum(P(zi | zi-1) * a(zi-1))
                fwd_sum = 0
                for z_previous in forward_messages[i - 1]:
                    # P(zi | zi-1) is named as P_zi_cond_ziprev
                    # alpha(zi-1)) is named as a_prev
                    P_zi_cond_ziprev = transition_model(z_previous)[zi]
                    a_prev = forward_messages[i - 1][z_previous]
                    fwd_sum = fwd_sum + P_zi_cond_ziprev * a_prev
                if fwd_sum != 0:
                    forward_messages[i][zi] = P_obs_cond_zi * fwd_sum
            # end of for loop to calculate a(zi)
            forward_messages[i].renormalize()
            # end of calculating forward message

    # TODO: Compute the backward messages
    for i in range(0, num_time_steps):
        last = num_time_steps - 1
        backward_messages[last - i] = rover.Distribution()
        if i == 0:  # initial state
            # beta (z_N-1) = 1
            for zi in all_possible_hidden_states:
                backward_messages[last - i][zi] = 1;
            backward_messages[last - i].renormalize()

        else:  # recursive part
            # (obs_xi+1, obs_yi+1) is named as observe_xiyi_plus1
            observe_xiyi_plus1 = observations[last - i + 1]
            for zi in all_possible_hidden_states:
                # sum of [ beta(z_i+1) * P(z_i+1 | zi) * P((obs_xi+1, obs_yi+1) | Z_i+1) ]
                bkw_sum = 0  # backwards sum
                for zi_aft in backward_messages[last - i + 1]:
                    # TODO: add part 2 here
                    if observe_xiyi_plus1 == None:
                        P_obs_cond_zi_plus1 = 1
                    else:
                        P_obs_cond_zi_plus1 = observation_model(zi_aft)[observe_xiyi_plus1]

                    beta_zi_plus1 = backward_messages[last - i + 1][zi_aft]
                    P_Ziplus1_cond_Zi = transition_model(zi)[zi_aft]
                    bkw_sum = bkw_sum + (beta_zi_plus1 * P_Ziplus1_cond_Zi * P_obs_cond_zi_plus1)
                if bkw_sum != 0:
                    backward_messages[last - i][zi] = bkw_sum
                #backward_messages[last-i].renormailize()

    # TODO: Compute the marginals
    for i in range(0, num_time_steps):
        marginals[i] = rover.Distribution()
        for zi in all_possible_hidden_states:
            alpha = forward_messages[i][zi]
            beta = backward_messages[i][zi]
            marginals[i][zi] = alpha * beta
        marginals[i].renormalize()

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    w = [None] * num_time_steps

    # first, find initial state; similar to forward message
    # ln (P(z0) * P(obs_x0, obs_y0 | z0))
    for i in range(0, num_time_steps):
        w[i] = rover.Distribution()
        if i == 0:
            observe_x0y0 = observations[0]
            for zi in prior_distribution:  # position represent (xn, yn)
                # w(z0) = P(z0) * P((x0, y0)|z0)
                P_z0 = prior_distribution[zi]
                P2 = observation_model(zi)[observe_x0y0]
                if P_z0 * P2 != 0:
                    w[i][zi] = np.log(P_z0 * P2)
            w[i].renormalize()

        else:
            observe_xiyi = observations[i]
            for zi in all_possible_hidden_states:
                # TODO: add part 2 here
                if observe_xiyi == None:
                    P_obs_cond_zi = 1
                else:
                    # alpha(zi) = P((xi_obs, yi_obs) | zi) * sum(P(zi | zi - 1) * alpha(zi - 1))
                    # P((xi_obs, yi_obs) | zi) is named P_obs_cond_zi
                    P_obs_cond_zi = observation_model(zi)[observe_xiyi]

                if P_obs_cond_zi != 0:
                    max_value = -99999999
                    for z_previous in w[i - 1]:
                        # P(zi | zi-1) is named as P_zi_cond_ziprev
                        # w(zi-1)) is named as w_prev
                        P_zi_cond_ziprev = transition_model(z_previous)[zi]
                        w_prev = w[i - 1][z_previous]
                        temp = np.log(P_zi_cond_ziprev) + w_prev
                        if temp > max_value:
                            max_value = temp

                    w[i][zi] = np.log(P_obs_cond_zi) * max_value
                # end of for loop to calculate a(zi)
            w[i].renormalize()
            # end of calculating w

        estimated_hidden_states[num_time_steps-1-i] = w[i][zi]

    return estimated_hidden_states


if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')



    # timestep = 30
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # TODO: part 4, calculate the error
    err_FB_count = 0
    marginals_len = len(marginals)
    for i in range(marginals_len):
        FB_state = marginals[i].get_mode()
        if FB_state == hidden_states[i]:
            err_FB_count += 1
    err_FB = 1 - err_FB_count/100
    print("error of forward_backward is: ", err_FB)

    err_Viterbi_count = 0
    est_state_len = len(estimated_states)
    for i in range(est_state_len):
        if(estimated_states[i] == hidden_states[i]):
            err_Viterbi_count += 1
    err_Viterbi = 1 - err_Viterbi_count/100
    print("error of Viterbi is: ", err_Viterbi)


    #TODO: part 5, find invalid move
    for i in range (1, num_time_steps-1):
        step = i
        x = marginals[i].get_mode()[0]
        y = marginals[i].get_mode()[1]
        a = marginals[i].get_mode()[2]
        if a == 'left':
            x += 1
        elif a == 'right':
            x -= 1
        elif a == 'up':
            y += 1
        elif a == 'down':
            y -= 1

        x_prev = marginals[i-1].get_mode()[0]
        y_prev = marginals[i-1].get_mode()[1]
        if x!=x_prev or y!=y_prev:
            print(step)
    # the result for this is step 65, so show the details for step 65 and 64
    print(64, marginals[64].get_mode())
    print(65, marginals[65].get_mode())

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

