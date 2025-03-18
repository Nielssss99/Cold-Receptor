import sqlite3
import matplotlib.pyplot as plt
from Olivares import solve_model
import ast

conn = sqlite3.connect("results.db")
c = conn.cursor()

# Show stored results
# Query to get the row with the lowest fitness (error)
c.execute("""
    SELECT * FROM results
    ORDER BY fitness ASC
    LIMIT 1;
""")

# Fetch the result
row = c.fetchone()
print("Frequency: ", row[3])
frqs = solve_model([0, 20, 25, 30, 35, 40], ast.literal_eval(row[2]))
print("Frequency Rerun: ", frqs)
plt.plot([20, 25, 30, 35, 40], frqs, label="Rerun")
plt.show()

conn.close()