

def update_item_increment_value(table, _key):
    response = table.update_item(
        Key={"email": "test1@example.com"},
        UpdateExpression="set total_orders = total_orders + :n",
        ExpressionAttributeValues={
            ":n": 1,
        },
        ReturnValues="UPDATED_NEW",
    )
    print(response["Attributes"])


def update_item_exiting_attribute(
        table,
        _key,
        attribute_to_update,
        value_to_update
        ):
    response = table.update_item(
        Key=_key,
        UpdateExpression=f"set #{attribute_to_update} = :n",
        ExpressionAttributeNames={
            f"#{attribute_to_update}": f"{attribute_to_update}",
        },
        ExpressionAttributeValues={
            ":n": f"{value_to_update}",
        },
        ReturnValues="UPDATED_NEW",
    )
    print(response["Attributes"])
