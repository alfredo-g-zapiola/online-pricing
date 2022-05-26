import numpy as np



class SocialInfluence:
    def __init__(self, todays_users):
        # todays_users looks like this: [100 200 150]
        self.todays_users= todays_users

        # TODO creare qui 3 graphs: 3 init prob matrices created with uniform
        # cardinalita di ogni graph e' il numero di users
        self.graphs = [graph1, graph2, graph3]
        # DONE
        self.n_steps_max = 10  # da definire in base ai costumers
        # Create three graphs accoriding to the customer

    # group number array : [ 1 ng1
    #                      2 ng2
    #                      3 ng3 ] only one of the 3 each iteration

    def simulate_influence(self, init_data):
        """
        Only public method oltre al constructor
        :param init_data: list of tuples with (index customer, product, 1 bought 0 did not buy)
        {
            "g1":[(41, 2, 0), (120, 1, 1)],
            "g2": [(150, 5, 1)],
            "g3!
        :return:
        """

        for g in range(3):  # per ogni gruppo
            init_data_g = init_data["g" + str(g + 1)]
            for p in range(5):  # per ogni prodotto
                pass
        # . Make the matrices unbalanced taking experience of clients of this group for this product
        # apply estimated prob
        # monte carlo
        # save for every user in this group, the influence to buy this product

        # finiscono i loop
        # fra quelli che non si trovavano in init_data:
        # si sceglie come prodotto la prob influence massima, con il numero di prodotto

        # costruire newly_active customers:
        # [(index cliente, prodotto, probabilita')]
        newly_active_customers = [(19, 3, .7), (76, 5, .1)]
        return newly_active_customers

    def __simulate_influence_graph(self, init_prob_matrix, n_steps_max, initdata, group_number):
        """
        NB __ lo rende privato
        Cosa si fa in questa funzine
        :param init_prob_matrix:
        :param n_steps_max:
        :return:
        """
        # simulazione active graph influenza con n step di influenza
        prob_matrix = init_prob_matrix.copy()
        current_customers = prob_matrix.shape[0]
        # initial active customers li fornisce alfredo
        initial_active_customers = np.random.binomial(1, 0.1, size=current_customers)
        history = np.array([initial_active_customers])
        active_customers = initial_active_customers
        newly_active_customers = active_customers
        init_prob_matrix = np.random.uniform(
            0, 0.1, (number_clients.shape[0], number_clients.shape[0])
        )
        matrix_unbalanced = probability_matrix_unbalanced(
            init_prob_matrix, newly_active_customers, group_number
        )
        active_edges = np.zeros(matrix_unbalanced.shape[0], matrix_unbalanced.shape[1])

        t = 0
        graph = np.zeros((matrix_unbalanced.shape[0], matrix_unbalanced.shape[0]))
        for i in initial_active_customers:
            if initial_active_customers[i] != 0 & graph[i, i] == 0:
                graph[i, i] = +1  # diventa 1 la diag del cliente attivo

        # creazione active graph con n_steps_max iterazioni
        while t < n_steps_max and np.sum(newly_active_customers) > 0:
            p = (matrix_unbalanced.T * active_customers).T
            # introdurre condizione per tracciare l'influenza tra i nodi
            # inserire ciclo di controllo indicizzato per salvare chi ha influenzato chi
            for i in p:
                influence_check = np.random.rand(p.shape[0], p.shape[0])
                for j in range(i, influence_check.shape[0]):
                    active_edges[i, j] = (
                        p[i] > influence_check[i, j]
                    )  # servono if #cambiare dim np.rand
                    if graph[i, i] == 0 & graph[i, j] == 0:
                        graph[i, i] = +1
                        graph[i, j] = +1  # diventa 1 la diag del cliente attivo

            matrix_unbalanced = matrix_unbalanced * ((p != 0) == active_edges)
            newly_active_customers = (np.sum(active_edges, axis=0) > 0) * (1 - active_customers)
            active_customers = np.array(active_customers + newly_active_customers)
            history = np.concatenate((history, [newly_active_customers]), axis=0)
            t += 1
        # serve dictionary che  contenga i 3 graph
        return struc  # registra active graph

    def __probability_matrix_unbalanced(
        self, init_prob_matrix, initdata, group_number, number_clients
    ):
        # cliente non  compra bassa probo
        # init data e' viene come [array_client, array_quale_prodtto, ary_rating]
        # if rating is 5, add 20% to the random influence (in probability matrix)
        # if rating is 4, add 10%
        # if rating is 3, just the random one
        # if rating is 2 or 1, the initial client is no longer a seed
        unbalanced_prob_matrix = init_prob_matrix
        initdata[:][1] = (
            initdata[:][1]
            - (group_number == 2) * ng1 * np.array(initdata.shape[0])
            - (group_number == 3) * (ng1 + ng2) * np.array(initdata.shape)
        )

        for i in initdata[:][1]:
            for j in range(number_clients):
                if initdata[j][3] == 5:
                    unbalanced_prob_matrix[j][:] = unbalanced_prob_matrix[j][:] * 1.2
                if initdata[j][3] == 4:
                    unbalanced_prob_matrix[j][:] = unbalanced_prob_matrix[j][:] * 1.1
                if initdata[j][3] == 3:
                    unbalanced_prob_matrix[j][:] = unbalanced_prob_matrix[j][:] * 1
                if initdata[j][3] == 2 or initdata[j][3] == 1:
                    unbalanced_prob_matrix[j][:] = unbalanced_prob_matrix[j][:] * 0
                    # implementare controllo probo !=1;
        return unbalanced_prob_matrix

    def __estimate_probabilities(self, dataset, node_index, n_nodes):
        # dataset = collezione delle history
        estimated_prob = np.ones(n_nodes) * 1.0 / (n_nodes - 1)
        credits = np.zeros(n_nodes)
        occurr_v_active = np.zeros(n_nodes)
        n_episodes = len(dataset)  # numero di grafi simulati
        for episode in dataset:
            idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
            if len(idx_w_active) > 0 and idx_w_active > 0:
                active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
                credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
            for v in range(0, n_nodes):
                if v != node_index:
                    idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                    if len(idx_v_active) > 0 and (
                        idx_v_active < idx_w_active or len(idx_w_active) == 0
                    ):
                        occurr_v_active[v] += 1
        estimated_prob = credits / occurr_v_active
        estimated_prob = np.nan_to_num(estimated_prob)
        return estimated_prob

    def simulate_influence_history(self, n_steps_max, direct_clients, group_number, initdata):
        """
        function to simulate the influence succession without considering the direct influence-graph
        in order to proper visualize the number of new clients after a number of influence iteration.

        :param n_steps_max: maximum number of iteration
        :param direct_clients: dictionary containing all the 3 groups of clients to simulate
        :param group_number: flag of the  group you're intrested in
        :return history: chronology of the progressive newly active customer by simple influence
        """
        number_clients = initdata.shape(1)
        init_prob_matrix = np.random.uniform(
            0, 0.1, (number_clients.shape[0], number_clients.shape[0])
        )
        prob_matrix = probability_matrix_unbalanced(
            self, init_prob_matrix, initdata, group_number, number_clients
        )
        ## n_customers = prob_matrix.shape[0]
        if group_number == 1:
            direct_clients_it = direct_clients.items("group_1")
        elif group_number == 2:
            direct_clients_it = direct_clients.items("group_2")
        else:
            direct_clients_it = direct_clients.items("group_3")

        customers_list = list(direct_clients_it)
        initial_active_customers = np.array(customers_list)
        history = np.array([initial_active_customers])
        active_customers = initial_active_customers
        newly_active_customers = np.array([])
        t = 0
        while t < n_steps_max and np.sum(newly_active_customers) > 0:
            p = (prob_matrix.T * active_customers).T
            active_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p != 0) == active_edges)
            newly_active_customers = (np.sum(active_edges, axis=0) > 0) * (1 - active_customers)
            active_customers = np.array(active_customers + newly_active_customers)
            history = np.concatenate((history, [newly_active_customers]), axis=0)
            t += 1
        return history


# n_nodes = 5
# n_episodes = 1000
# prob_matrix = np.random.uniform(0, 0.1, (n_nodes, n_nodes))
# node_index = 4
# dataset = []
# for i in range(0, n_episodes):
#  dataset.append(simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=10))
#
# estimated_prob = estimate_probabilities(dataset=dataset, node_index=node_index, n_nodes=n_nodes)
