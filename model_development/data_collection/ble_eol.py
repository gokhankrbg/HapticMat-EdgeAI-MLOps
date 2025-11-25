import asyncio
import struct
from bleak import BleakClient

DEVICE_ADDRESS = "44CED45D-83B0-BC60-BE09-4F76D13DE480"  # PCB B x11111
TX_CHARACTERISTIC_UUID = "6e400002-c352-11e5-953d-0002a5d5c51b"  
RX_CHARACTERISTIC_UUID = "6e400003-c352-11e5-953d-0002a5d5c51b"

PAYLOAD = bytearray([0x01, 0xC0, 0x01])  # Example send payload

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        await client.write_gatt_char(TX_CHARACTERISTIC_UUID, bytearray([0x01, 0xD0, 0x01]))
        # print(f"Sent: {PAYLOAD.hex()}")
        await asyncio.sleep(1)

        def notification_handler(sender, data: bytearray):
            # print(f"Notification from {sender}: {data.hex()}")
            
            # Convert received payload to 4 floats if it has enough bytes
            if len(data) >= 16:  # 4 floats × 4 bytes
                float_values = struct.unpack('<4f', data[1:])
                print(f"Received as floats: {float_values}")
            else:
                print(f"Received payload too short to unpack 4 floats")

        await client.start_notify(RX_CHARACTERISTIC_UUID, notification_handler)

        try:
            while True:
                await client.write_gatt_char(TX_CHARACTERISTIC_UUID, PAYLOAD)
                # print(f"Sent: {PAYLOAD.hex()}")
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping notifications...")
        finally:
            await client.stop_notify(RX_CHARACTERISTIC_UUID)
            print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
