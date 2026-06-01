from operator import itemgetter


def greedy_batch_scheduler(orders):
    """
    Greedy Interval Scheduling for multi-order batching.

    Parameters
    ----------
    orders : list of tuples
        (order_id, ready_time, deadline)

    Returns
    -------
    dict containing:
        batches
        num_batches
        batch_size
    """

    # Earliest Deadline First (EDF)
    orders = sorted(orders, key=itemgetter(2))

    batches = []
    current_batch = []

    current_start = None
    current_end = None

    for order in orders:
        order_id, ready_time, deadline = order

        if not current_batch:
            current_batch.append(order)
            current_start = ready_time
            current_end = deadline
            continue

        new_start = max(current_start, ready_time)
        new_end = min(current_end, deadline)

        if new_start <= new_end:
            current_batch.append(order)
            current_start = new_start
            current_end = new_end
        else:
            batches.append(current_batch)

            current_batch = [order]
            current_start = ready_time
            current_end = deadline

    if current_batch:
        batches.append(current_batch)

    total_orders = len(orders)
    num_batches = len(batches)

    avg_batch_size = (
        round(total_orders / num_batches, 2)
        if num_batches > 0 else 0
    )

    return {
        "batches": batches,
        "num_batches": num_batches,
        "batch_size": avg_batch_size
    }


if __name__ == "__main__":

    orders = [
        ("O1", 10, 20),
        ("O2", 12, 22),
        ("O3", 14, 24),
        ("O4", 30, 40),
        ("O5", 31, 42),
        ("O6", 32, 45)
    ]

    result = greedy_batch_scheduler(orders)

    print("Number of batches:", result["num_batches"])
    print("Average batch size:", result["batch_size"])

    print("\nBatches:")
    for i, batch in enumerate(result["batches"], start=1):
        ids = [order[0] for order in batch]
        print(f"Batch {i}: {ids}")