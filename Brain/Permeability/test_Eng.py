import jax
import sys # Import sys to handle potential errors better

print("Attempting to list JAX devices...")
try:
    devices = jax.devices()
    print("Successfully retrieved JAX devices:")
    print(devices)
    print("\nScript finished successfully.")
except Exception as e:
    print(f"\nAn error occurred while calling jax.devices(): {e}", file=sys.stderr)
    # You might want to print more detailed traceback here if needed
    # import traceback
    # traceback.print_exc()
    sys.exit(1) # Exit with a non-zero code to indicate error

sys.exit(0) # Explicitly exit with success code