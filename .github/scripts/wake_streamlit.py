import os 
import sys
import asyncio
from playwright.async_api import async_playwright

# Get the app URL from environment variable (set in GitHub Secrets)
URL = os.getenv("STREAMLIT_URL")

if not URL:
    print("ERROR: STREAMLIT_URL environment variable is not set.")
    sys.exit(1)

async def wake_app():
    print(f"Visiting: {URL}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            # Navigate and wait for the page to load
            await page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(5000)  # Extra wait for any dynamic content

            # Look for the wake-up button
            wake_button = page.get_by_role("button", name="Yes, get this app back up!")

            if await wake_button.count() > 0:
                print("App is sleeping. Clicking wake button...")
                await wake_button.click()
                # Wait for the app to finish waking (can take 30–90 seconds)
                await page.wait_for_timeout(60_000)
                print("Wake button clicked. App should now be starting.")
            else:
                print("App is already awake (or wake button not found).")

            # Optional: take a screenshot for debugging (visible in Action logs as artifact if configured)
            # await page.screenshot(path="wake_result.png")

        except Exception as e:
            print(f"Error while waking app: {e}")
            # Don't fail the whole workflow on transient errors
            sys.exit(0)
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(wake_app())
