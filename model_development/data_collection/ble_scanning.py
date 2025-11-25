import asyncio
from bleak import BleakScanner


async def scan_devices(scan_time: float = 5.0):
    print(f"Starting BLE scan for {scan_time} sec...")
    devices = await BleakScanner.discover(timeout=scan_time, return_adv=True)
    print(f"Scan complete. {len(devices)} devices found:")
    for idx, (address, (device, adv_data)) in enumerate(devices.items(), 1):
        name = device.name or "<Unknown>"
        rssi = adv_data.rssi
        print(f"{idx:3}. {address} | RSSI: {rssi:4} dBm | {name}")
    return devices


async def run_continuous(scan_time: float = 5.0, interval: float = 10.0):
    while True:
        await scan_devices(scan_time)
        print(f"Waiting {interval} sec before next scan...")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    try:
        asyncio.run(scan_devices(scan_time=5.0))
    except KeyboardInterrupt:
        print("Scan interrupted by user")