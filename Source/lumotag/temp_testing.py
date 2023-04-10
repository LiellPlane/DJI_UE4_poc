import rabbit_mq
import factory
import messaging

def main():
    mssger = rabbit_mq.messenger(factory.TZAR_config())

if __name__ == '__main__':
    main()
