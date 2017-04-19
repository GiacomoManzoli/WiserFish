

from util.file_helper import save_all, load_all
from generator.generators import generate_dataset


# TODO: quantities

clients, products, orders, model = generate_dataset(clients_count=2,
                                                    products_count=10,
                                                    days_count=20,
                                                    day_interval=4,
                                                    model_name='cond')

save_all(clients, products, orders, model, 'test_')
