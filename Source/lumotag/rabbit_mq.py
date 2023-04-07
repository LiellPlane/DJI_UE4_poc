import factory
import pika
import logging
import threading
import time


def thread_function(name):
    credentials = pika.PlainCredentials(username='guest', password='guest')
    parameters = pika.ConnectionParameters(host='lumotagHQ.local',
                                        port=5672,
                                        virtual_host='/',
                                        credentials=credentials)
    connection = pika.BlockingConnection(parameters)

    channel = connection.channel()

    channel.exchange_declare(exchange='hits', exchange_type='fanout')

    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    channel.queue_bind(exchange='hits', queue=queue_name)

    print(' [*] Waiting for logs. To exit press CTRL+C')

    def callback(ch, method, properties, body):
        print("received [x] %r" % body)

    channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)

    channel.start_consuming()


if __name__ == "__main__":
    x = threading.Thread(target=thread_function, args=(1,))
    x.start()

    while True:
        time.sleep(2)
        print("plops for tea")












import pika

# common to producer and receiver
credentials = pika.PlainCredentials(username='guest', password='guest')
parameters = pika.ConnectionParameters(host='lumotagHQ.local',
                                    port=5672,
                                    virtual_host='/',
                                    credentials=credentials)
connection = pika.BlockingConnection(parameters)

channel = connection.channel()

channel.exchange_declare(exchange='hits',
                         exchange_type='fanout')

#end of common block

result = channel.queue_declare(queue='', exclusive=True)
channel.queue_bind(exchange='hits',
                   queue=result.method.queue)

message = "farts 4 uuu"
channel.basic_publish(exchange='hits', routing_key='', body=message)
print(" [x] Sent %r" % message)
connection.close()
