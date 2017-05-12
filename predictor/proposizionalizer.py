import pandas as pd
from datetime import datetime


def proposizionalize(orders, clients, products):
    """
    :param orders: dictionary indexed by date (timestamp format) containing the orders binary matrix for the day
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
                order_date = datetime.fromtimestamp(key)
                day_of_year = order_date.timetuple().tm_yday
                year = order_date.timetuple().tm_year
                orders_rows.append({
                    'datetime': key,  # timestamp of the key (order's date)
                    'day_of_year': day_of_year,
                    'year': year,
                    'clientId': clients.iloc[c]['clientId'],
                    'productId': products.iloc[p]['productId'],
                    'ordered': matrix[c, p]
                })

    orders_df = pd.DataFrame(orders_rows, columns=['datetime', 'day_of_year', 'year', 'clientId', 'productId', 'ordered'])

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
