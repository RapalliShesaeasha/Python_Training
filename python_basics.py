def calculate_final_price(price: float, discount: float) -> float:
    if price < 0:
        raise ValueError("Price cannot be negative")

    if not 0 <= discount <= 1:
        raise ValueError("Discount must be between 0 and 1")

    final_price = price - (price * discount)
    return final_price


class User:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role.lower()

    def is_admin(self) -> bool:
        return self.role == "admin"


if __name__ == "__main__":
    # Function test
    print("Final Price:", calculate_final_price(500, 0.2))

    # Class test
    user1 = User("Sheshu", "Admin")
    user2 = User("Adrija", "Viewer")

    print(f"{user1.name} is admin:", user1.is_admin())
    print(f"{user2.name} is admin:", user2.is_admin())
