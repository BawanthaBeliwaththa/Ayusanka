from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        
        print("Navigating to app...")
        page.goto("http://localhost:8000")
        page.wait_for_timeout(3000)
        
        print("Capturing Screenshot 1: Individual Predictor...")
        page.screenshot(path="screenshot_1_predictor.png", full_page=True)
        
        print("Switching to National Forecasting tab...")
        page.click('.tab-btn[data-tab="national"]')
        page.wait_for_timeout(2000)
        page.screenshot(path="screenshot_2_national.png", full_page=True)
        
        print("Switching to Policy Simulator tab...")
        page.click('.tab-btn[data-tab="policy"]')
        page.wait_for_timeout(2000)
        page.screenshot(path="screenshot_3_policy.png", full_page=True)
        
        print("Switching to Model & Insights tab...")
        page.click('.tab-btn[data-tab="academic"]')
        page.wait_for_timeout(2000)
        page.screenshot(path="screenshot_4_academic.png", full_page=True)
        
        print("Switching language to Sinhala...")
        page.click('#lang-toggle-btn')
        page.wait_for_timeout(1000)
        page.click('.tab-btn[data-tab="predictor"]')
        page.wait_for_timeout(2000)
        page.screenshot(path="screenshot_5_sinhala.png", full_page=True)

        browser.close()
        print("All screenshots captured successfully!")

if __name__ == "__main__":
    run()
