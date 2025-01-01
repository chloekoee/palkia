import threading
import time

def poll_market_data(client, frequency = 10):
    while True:
        order_book = client.get_order_book()
        ### perform analysis on the order book
        time.sleep(frequency)

def poll_weather_data(model, frequency = 1800):
    while True:
        ### poll for new weather data from bom (not exchange) 
        ### upon receiving weather data, update model 
        ### update order 
        pass

 
'''
If all signals are handled by the client class, then whilst the client 
is processing one signal 
'''