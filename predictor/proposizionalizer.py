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

    orders_df = pd.DataFrame(orders_rows,
                             columns=['datetime', 'day_of_year', 'year', 'clientId', 'productId', 'ordered'])

    orders_df['datetime'] = orders_df['datetime'].astype(dtype=long)
    orders_df['day_of_year'] = orders_df['day_of_year'].astype(dtype=int)
    orders_df['year'] = orders_df['year'].astype(dtype=int)
    orders_df['clientId'] = orders_df['clientId'].astype(dtype=int)
    orders_df['productId'] = orders_df['productId'].astype(dtype=int)
    orders_df['ordered'] = orders_df['ordered'].astype(dtype=int)

    # 2. Join the Dataframes
    orders_df = orders_df.join(clients, on='clientId', lsuffix='_o', rsuffix='_c')
    orders_df = orders_df.join(products, on='productId', lsuffix='_o', rsuffix='_p')

    orders_df['client_id'] = orders_df['clientId_o']
    orders_df['product_id'] = orders_df['productId_o']
    orders_df = orders_df.drop(['clientId_o', 'clientId_c', 'productId_p', 'productId_o'], axis=1)

    # 3. Clears extra data
    orders_df = orders_df.drop('client_name', axis=1)
    orders_df = orders_df.drop('product_name', axis=1)

    orders_df = orders_df.drop('pc', axis=1)
    orders_df = orders_df.drop('client_freq_scale', axis=1)
    orders_df = orders_df.drop('pp', axis=1)
    orders_df = orders_df.drop('product_freq_scale', axis=1)

    return orders_df
