import factory
import pika
import time
import messaging
from dataclasses import asdict
from socket import gaierror
import msgs
import json
import uuid

class RabbitMQ_Obj():
    def __init__(
            self,
            messaging_config,
            name,
            send_only):# dict[str, str]) -> None:

        mc = messaging_config
        credentials = pika.PlainCredentials(
            username=mc["username"],
            password=mc["password"])

        parameters = pika.ConnectionParameters(
            host=mc["host"],
            port=mc["port"],
            virtual_host=mc["virtual_host"],
            credentials=credentials
            )

        while True:
            try:
                self.connection = pika.BlockingConnection(parameters)
                break
            except gaierror as e:
                print("rabbitqm Err, Is message server active?")

            time.sleep(5)
            
        self.channel = self.connection.channel()

        self.channel.exchange_declare(
            exchange='hits',
            exchange_type='fanout')

        if not send_only:
            result = self.channel.queue_declare(
                queue=name,
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


class Messenger(factory.Messenger):

    def __init__(self, config) -> None:
        super().__init__(config=config)

    def _heartbeat(self, out_box, config):
        while True:
            time.sleep(config.msg_heartbeat_s)
            hb = msgs.create_heartbeat_msg(config)
            out_box.put(
                hb,
                block=True)

    def _in_box_worker(self, in_box, config, scheduler):
        msg_worker = RabbitMQ_Obj(
            config.messaging_config,
            f"{config.my_id}_consumer",
            send_only=False)
        callback_hndl = CallBack_QueueHandler(in_box, config)
        scheduler.put("IN BOX READY")
        msg_worker.start_consuming(callback_hndl.callback_handler)

    def _out_box_worker(self, out_box, config, scheduler):
        # will block until InBox worker has started its RMQ connection
        # then in theory should be ready to init another
        scheduler.get(block=True)
        # arbitrary sleep needed even though we have confirmed
        # first connection has been initialised
        # TODO must be some way to avoid this
        time.sleep(3)
        msg_worker = RabbitMQ_Obj(
            config.messaging_config,
            f"{config.my_id}_sender",
            send_only=True)
        time.sleep(1)
        # we can assume that consumer connection is active
        # send a message that we are ready
        hello_msg = msgs.package_dataclass_to_bytes(msgs.Report(
                my_id=config.my_id,
                target="",
                timestamp=msgs.get_epoch_ts(),
                img_as_str=None,
                msg_type=msgs.MessageTypes.HELLO.value,
                msg_string=""
            ))
        
        msg_worker.send_message(hello_msg)
        out_box.queue.clear()
        while True:
          message = out_box.get(block=True)
          msg_worker.send_message(message)

class MessengerBasic(factory.Messenger):
    
    def __init__(self, config) -> None:
        """Basic version with no heartbeat and
        connection messages"""
        super().__init__(config=config)

    def _heartbeat(self, out_box, config):
        pass

    def _in_box_worker(self, in_box, config, scheduler):
        msg_worker = RabbitMQ_Obj(
            config.messaging_config,
            f"{config.my_id}_consumer",
            send_only=False)
        callback_hndl = CallBack_QueueHandler(in_box, config)
        scheduler.put("IN BOX READY")
        msg_worker.start_consuming(callback_hndl.callback_handler)

    def _out_box_worker(self, out_box, config, scheduler):
        # will block until InBox worker has started its RMQ connection
        # then in theory should be ready to init another
        scheduler.get(block=True)
        # arbitrary sleep needed even though we have confirmed
        # first connection has been initialised
        # TODO must be some way to avoid this
        time.sleep(3)
        msg_worker = RabbitMQ_Obj(
            config.messaging_config,
            f"{config.my_id}_sender",
            send_only=True)
        time.sleep(1)

        while True:
          message = out_box.get(block=True)
          msg_worker.send_message(message)

class CallBack_QueueHandler():
    # callback is needed with the in box queue,
    # so create it as a class and give the rabbit callback
    # as a method in the class with access to members we need
    def __init__(self, inbox, config) -> None:
        """Class to handle callbacks from rabbitMQ and placing them in
        a queue for the main thread to handle"""
        self._in_box = inbox
        self._gunconfig = config

    def create_qfull_err(self, err_msg):
        msg_to_send = msgs.Report(
            my_id=self._gunconfig.my_id,
            target=None,
            timestamp=msgs.get_epoch_ts(),
            img_as_str=None,
            msg_type=msgs.MessageTypes.ERROR.value,
            msg_string=err_msg
        )

        return msgs.package_dataclass_to_bytes(msg_to_send)

    def callback_handler(self, ch, method, properties, body):
        if self._in_box._qsize() >= self._in_box.maxsize:
            err_msg = "Incoming int queue full! All unhandled msgs cleared. "
            err_msg += f"qsize of {self._in_box._qsize()}, max of {self._in_box.maxsize}"
            self._in_box.queue.clear()
            self._in_box.put(
                self.create_qfull_err(err_msg),
                block=False)
            return

        self._in_box.put(
            body,
            block=False)
