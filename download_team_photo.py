"""Download team photo for the application"""
import urllib.request

url = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200&q=80"
output_path = "assets/team_photo.jpg"

print(f"Downloading medical team photo...")
urllib.request.urlretrieve(url, output_path)
print(f"✓ Photo saved to {output_path}")
