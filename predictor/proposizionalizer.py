import pandas as pd


def proposizionalize(orders, clients, products):
    """

    :param orders: dictionary indexed by date (timestamp format) containing the order matrix of the day
    :param clients: dataframe with clients' data
    :param products: dataframe with products' data
    :return: proposizionalized dataframe
    """

    # 1. Create a Dataframe for the orders from the matrices
    #    The DF must contain even the "not-ordered row"
    orders_rows = []
    for key in orders.keys():
        matrix = orders[key]
        clients_count = matrix.shape[0]
        products_count = matrix.shape[1]
        for c in range(0, clients_count):
            for p in range(0, products_count):
                orders_rows.append({
                    'datetime': key,  # timestamp of the key (order's date)
                    'clientId': clients.iloc[c]['clientId'],
                    'productId': products.iloc[p]['productId'],
                    'ordered': matrix[c, p]
                })
    orders_df = pd.DataFrame(orders_rows, columns=['datetime', 'clientId', 'productId', 'ordered'])

    # 2. Join the Dataframes
    orders_df = orders_df.join(clients, on='clientId', lsuffix='_o', rsuffix='_c')
    orders_df = orders_df.join(products, on='productId', lsuffix='_o', rsuffix='_p')

    # 3. Clears extra data
    orders_df = orders_df.drop('client_name', axis=1)
    orders_df = orders_df.drop('product_name', axis=1)

    orders_df = orders_df.drop('pc', axis=1)
    orders_df = orders_df.drop('client_freq_scale', axis=1)
    orders_df = orders_df.drop('pp', axis=1)
    orders_df = orders_df.drop('product_freq_scale', axis=1)

    return orders_df
