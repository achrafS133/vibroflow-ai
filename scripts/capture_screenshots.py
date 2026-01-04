"""
Screenshot capture script for VibroFlow AI Dashboard
Captures all tabs for README documentation
"""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import time

DASHBOARD_URL = "http://localhost:8503"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "screenshots"


async def capture_screenshots():
    """Capture screenshots of all dashboard tabs."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=2  # High DPI for crisp images
        )
        page = await context.new_page()
        
        print("Connecting to dashboard...")
        await page.goto(DASHBOARD_URL, wait_until='networkidle')
        await asyncio.sleep(3)  # Wait for charts to render
        
        # Screenshot 1: Main dashboard (Real-Time Monitoring)
        print("Capturing: Real-Time Monitoring...")
        await page.screenshot(
            path=str(OUTPUT_DIR / "01_realtime_monitoring.png"),
            full_page=False
        )
        
        # Screenshot 2: Vibration Analysis Tab
        print("Capturing: Vibration Analysis...")
        tabs = page.locator('[data-baseweb="tab"]')
        await tabs.nth(1).click()
        await asyncio.sleep(2)
        await page.screenshot(
            path=str(OUTPUT_DIR / "02_vibration_analysis.png"),
            full_page=False
        )
        
        # Screenshot 3: Flow Estimation Tab
        print("Capturing: Flow Estimation...")
        await tabs.nth(2).click()
        await asyncio.sleep(2)
        await page.screenshot(
            path=str(OUTPUT_DIR / "03_flow_estimation.png"),
            full_page=False
        )
        
        # Screenshot 4: Maintenance Prediction Tab
        print("Capturing: Maintenance Prediction...")
        await tabs.nth(3).click()
        await asyncio.sleep(2)
        await page.screenshot(
            path=str(OUTPUT_DIR / "04_maintenance_prediction.png"),
            full_page=False
        )
        
        await browser.close()
        
    print(f"\nScreenshots saved to: {OUTPUT_DIR}")
    print("Files created:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    asyncio.run(capture_screenshots())
