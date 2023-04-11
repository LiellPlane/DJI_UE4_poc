import factory
import pika
import time
import messaging
from dataclasses import asdict
from socket import gaierror


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

        try:
            self.connection = pika.BlockingConnection(parameters)
        except gaierror as e:
           print("rabbitqm Err, Is message server active?")
           raise e

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

    def start_consuming(
            self,
            call_back_function: callable):

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=call_back_function,
            auto_ack=True)
        # next command will block here until end of time
        # input messages will be directed to the callback
        # function & subsequent queue back to main thread
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

    def _in_box_worker(self, in_box, config, scheduler):
        # will block until OutBox worker has started its RMQ connection
        # then in theory should be ready to init another
        scheduler.get(block=True)
        # arbitrary sleep needed even though we have confirmed
        # first connection has been initialised
        # TODO must be some way to avoid this
        time.sleep(3)
        msg_worker = RabbitMQ_Obj(config.messaging_config)
        callback_hndl = CallBack_QueueHandler(in_box)
        msg_worker.start_consuming(callback_hndl.callback_handler)

    def _out_box_worker(self, out_box, config, scheduler):
        msg_worker = RabbitMQ_Obj(config.messaging_config)
        scheduler.put("OUT BOX READY")
        while True:
           message = out_box.get(block=True)
           msg_worker.send_message(message)


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
