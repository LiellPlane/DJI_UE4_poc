import factory
import pika
import logging
import threading
import time
import messaging
from dataclasses import asdict

class RabbitMQ_Connection():
    def __init__(
            self,
            messaging_config: dict[str, str]) -> None:

        mc = messaging_config
        credentials = pika.PlainCredentials(
            username=mc["username"],
            password=mc["password"])

        parameters = pika.ConnectionParameters(host=mc["host"],
                                            port=mc["port"],
                                            virtual_host=mc["virtual_host"],
                                            credentials=credentials)

        self.connection = pika.BlockingConnection(parameters)


class RabbitMQ_Obj():
    def __init__(
            self,
            messaging_config: dict[str, str]) -> None:

        mc = messaging_config
        credentials = pika.PlainCredentials(
            username=mc["username"],
            password=mc["password"])

        parameters = pika.ConnectionParameters(host=mc["host"],
                                            port=mc["port"],
                                            virtual_host=mc["virtual_host"],
                                            credentials=credentials)

        self.connection = pika.BlockingConnection(parameters)

        self.channel = self.connection.channel()

        self.channel.exchange_declare(
            exchange='hits',
            exchange_type='fanout')

        result = self.channel.queue_declare(
            queue='',
            exclusive=True)

        self.queue_name = result.method.queue

        self.channel.queue_bind(
            exchange='hits',
            queue=self.queue_name)
    print("rabbit MQ initied")
    def start_consuming(
            self,
            call_back_function: callable):

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=call_back_function,
            auto_ack=True)
        # next command will block here until end of time
        # input messages will be directed to the callback
        # function
        self.channel.start_consuming()
    
    def send_message(self, message):
        self.channel.basic_publish(
            exchange='hits',
            routing_key='',
            body=message)
        return


class messenger(factory.messenger):

    def __init__(self, config) -> None:
        super().__init__(config=config)

    def _in_box_worker(self, in_box):
        msg_worker = RabbitMQ_Obj(self._config.messaging_config)
        callback_hndl = CallBack_QueueHandler(in_box)
        msg_worker.start_consuming(callback_hndl.callback_handler)

    def _out_box_worker(self, out_box):
        msg_worker = RabbitMQ_Obj(self._config.messaging_config)
        while True:
            message = out_box.get(block=True)
            msg_worker.send_message(message)


def thread_function(name):
    credentials = pika.PlainCredentials(username='guest', password='guest')
    parameters = pika.ConnectionParameters(host='lumotagHQ.local',
                                        port=5672,
                                        virtual_host='/',
                                        credentials=credentials)
    connection = pika.BlockingConnection(parameters)

    channel = connection.channel()

    channel.exchange_declare(
        exchange='hits',
        exchange_type='fanout')

    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    channel.queue_bind(exchange='hits', queue=queue_name)

    print(' [*] Waiting for logs. To exit press CTRL+C')

    def callback(ch, method, properties, body):
        print("received [x] %r" % body)

    channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)

    channel.start_consuming()


class CallBack_QueueHandler():
    # callback is needed with the in box queue,
    # so create it as a class and give the rabbit callback
    # as a method in the class with access to members we need
    def __init__(self, inbox) -> None:
        """Class to handle callbacks from rabbitMQ and placing them in
        a queue for the main thread to handle"""
        self._in_box = inbox

    def callback_handler(self, ch, method, properties, body):
        if self._in_box._qsize() >= self._in_box.maxsize - 1:
            self._in_box.queue.clear()
            self._in_box.put(
                asdict(messaging.error(
                error_str="HIT REPORT INCOMING QUEUE FULL")),
                block=False)
            return

        self._in_box.put(
            str(body),
            block=False)

def send_msg():

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


if __name__ == "__main__":
    x = threading.Thread(target=thread_function, args=(1,))
    x.start()

    while True:
        time.sleep(1)
        print("plops for tea")
        send_msg()