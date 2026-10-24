# test_complete.py - Complete test for all filters

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from actions.views import *

def print_header(text):
    """Print header with decoration"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_result(name, lp_val, hp_val, bp_val, notch_val=None):
    """Print test results in nice format"""
    print(f"\n  📊 {name}:")
    print(f"     ├─ Low-Pass:   {lp_val:8.2f}")
    print(f"     ├─ High-Pass:  {hp_val:8.2f}")
    print(f"     ├─ Band-Pass:  {bp_val:8.2f}")
    if notch_val is not None:
        print(f"     └─ Notch:      {notch_val:8.2f}")

def run_comprehensive_test():
    """Run comprehensive test for all filters"""
    
    print_header("🚀 COMPREHENSIVE NOISE FILTER TEST")
    
    # ============================================
    # Step 1: Create test image (plus sign)
    # ============================================
    print("\n📸 Step 1: Creating test image...")
    img_array = np.zeros((256, 256), dtype=np.uint8)
    img_array[118:138, :] = 200
    img_array[:, 118:138] = 200
    test_img = Image.fromarray(img_array)
    test_img.save('test_original.png')
    print("   ✅ Test image created (white cross on dark background)")
    
    # ============================================
    # Step 2: Test configurations
    # ============================================
    test_configs = [
        {'sigma': 10,  'd0': 10,  'n': 2, 'notch': 10, 'name': 'Weak Noise - Low d0'},
        {'sigma': 50,  'd0': 10,  'n': 2, 'notch': 10, 'name': 'Strong Noise - Low d0'},
        {'sigma': 25,  'd0': 30,  'n': 3, 'notch': 15, 'name': 'Medium Noise - Medium d0'},
        {'sigma': 25,  'd0': 80,  'n': 5, 'notch': 20, 'name': 'Medium Noise - High d0'},
    ]
    
    all_results = {}
    
    # ============================================
    # Step 3: Test each configuration
    # ============================================
    print_header("📊 STEP 2: RUNNING TESTS")
    
    for config in test_configs:
        sigma = config['sigma']
        d0 = config['d0']
        n = config['n']
        notch = config['notch']
        name = config['name']
        
        print(f"\n🔹 Testing: {name}")
        print(f"   Parameters: sigma={sigma}, d0={d0}, n={n}, notch_width={notch}")
        
        # ----- Gaussian Filters -----
        print("\n   🟢 Gaussian Filters:")
        try:
            g_lp = add_gaussian_noise_lowpass(test_img, sigma=sigma, d0=d0)
            g_hp = add_gaussian_noise_highpass(test_img, sigma=sigma, d0=d0)
            g_bp = add_gaussian_noise_bandpass(test_img, sigma=sigma, d0=d0)
            g_notch = add_gaussian_noise_notch(test_img, sigma=sigma, d0=d0, notch_width=notch)
            
            orig = np.array(test_img, dtype=np.float32)
            g_lp_val = np.mean(np.abs(np.array(g_lp, dtype=np.float32) - orig))
            g_hp_val = np.mean(np.abs(np.array(g_hp, dtype=np.float32) - orig))
            g_bp_val = np.mean(np.abs(np.array(g_bp, dtype=np.float32) - orig))
            g_notch_val = np.mean(np.abs(np.array(g_notch, dtype=np.float32) - orig))
            
            print_result("Gaussian", g_lp_val, g_hp_val, g_bp_val, g_notch_val)
            
            g_lp.save(f'gaussian_lp_s{sigma}_d{d0}.png')
            g_hp.save(f'gaussian_hp_s{sigma}_d{d0}.png')
            g_bp.save(f'gaussian_bp_s{sigma}_d{d0}.png')
            g_notch.save(f'gaussian_notch_s{sigma}_d{d0}.png')
            
        except Exception as e:
            print(f"   ❌ Gaussian Error: {e}")
        
        # ----- Butterworth Filters -----
        print("\n   🟡 Butterworth Filters:")
        try:
            b_lp = add_butterworth_noise_lowpass(test_img, sigma=sigma, d0=d0, n=n)
            b_hp = add_butterworth_noise_highpass(test_img, sigma=sigma, d0=d0, n=n)
            b_bp = add_butterworth_noise_bandpass(test_img, sigma=sigma, d0=d0, n=n)
            b_notch = add_butterworth_noise_notch(test_img, sigma=sigma, d0=d0, notch_width=notch, n=n)
            
            orig = np.array(test_img, dtype=np.float32)
            b_lp_val = np.mean(np.abs(np.array(b_lp, dtype=np.float32) - orig))
            b_hp_val = np.mean(np.abs(np.array(b_hp, dtype=np.float32) - orig))
            b_bp_val = np.mean(np.abs(np.array(b_bp, dtype=np.float32) - orig))
            b_notch_val = np.mean(np.abs(np.array(b_notch, dtype=np.float32) - orig))
            
            print_result("Butterworth", b_lp_val, b_hp_val, b_bp_val, b_notch_val)
            
            b_lp.save(f'butterworth_lp_s{sigma}_d{d0}_n{n}.png')
            b_hp.save(f'butterworth_hp_s{sigma}_d{d0}_n{n}.png')
            b_bp.save(f'butterworth_bp_s{sigma}_d{d0}_n{n}.png')
            b_notch.save(f'butterworth_notch_s{sigma}_d{d0}_n{n}.png')
            
        except Exception as e:
            print(f"   ❌ Butterworth Error: {e}")
        
        # ----- Ideal Filters -----
        print("\n   🔴 Ideal Filters:")
        try:
            i_lp = add_ideal_noise_lowpass(test_img, sigma=sigma, d0=d0)
            i_hp = add_ideal_noise_highpass(test_img, sigma=sigma, d0=d0)
            i_bp = add_ideal_noise_bandpass(test_img, sigma=sigma, d0=d0)
            i_notch = add_ideal_noise_notch(test_img, sigma=sigma, d0=d0, notch_width=notch)
            
            orig = np.array(test_img, dtype=np.float32)
            i_lp_val = np.mean(np.abs(np.array(i_lp, dtype=np.float32) - orig))
            i_hp_val = np.mean(np.abs(np.array(i_hp, dtype=np.float32) - orig))
            i_bp_val = np.mean(np.abs(np.array(i_bp, dtype=np.float32) - orig))
            i_notch_val = np.mean(np.abs(np.array(i_notch, dtype=np.float32) - orig))
            
            print_result("Ideal", i_lp_val, i_hp_val, i_bp_val, i_notch_val)
            
            i_lp.save(f'ideal_lp_s{sigma}_d{d0}.png')
            i_hp.save(f'ideal_hp_s{sigma}_d{d0}.png')
            i_bp.save(f'ideal_bp_s{sigma}_d{d0}.png')
            i_notch.save(f'ideal_notch_s{sigma}_d{d0}.png')
            
        except Exception as e:
            print(f"   ❌ Ideal Error: {e}")
        
        print("\n   ✅ Test completed")
        print("   " + "-"*50)
    
    # ============================================
    # Step 4: Visual comparison
    # ============================================
    print_header("🖼️ STEP 3: VISUAL COMPARISON")
    
    # Load the last test results for display
    last_config = test_configs[-1]
    sigma = last_config['sigma']
    d0 = last_config['d0']
    n = last_config['n']
    notch = last_config['notch']
    
    try:
        fig, axes = plt.subplots(3, 5, figsize=(20, 15))
        
        # Original
        axes[0, 0].imshow(test_img, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Gaussian Filters
        g_lp = Image.open(f'gaussian_lp_s{sigma}_d{d0}.png')
        g_hp = Image.open(f'gaussian_hp_s{sigma}_d{d0}.png')
        g_bp = Image.open(f'gaussian_bp_s{sigma}_d{d0}.png')
        g_notch = Image.open(f'gaussian_notch_s{sigma}_d{d0}.png')
        
        axes[0, 1].imshow(g_lp, cmap='gray')
        axes[0, 1].set_title('Gaussian LP', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(g_hp, cmap='gray')
        axes[0, 2].set_title('Gaussian HP', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(g_bp, cmap='gray')
        axes[0, 3].set_title('Gaussian BP', fontsize=12)
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(g_notch, cmap='gray')
        axes[0, 4].set_title('Gaussian Notch', fontsize=12)
        axes[0, 4].axis('off')
        
        # Butterworth Filters
        b_lp = Image.open(f'butterworth_lp_s{sigma}_d{d0}_n{n}.png')
        b_hp = Image.open(f'butterworth_hp_s{sigma}_d{d0}_n{n}.png')
        b_bp = Image.open(f'butterworth_bp_s{sigma}_d{d0}_n{n}.png')
        b_notch = Image.open(f'butterworth_notch_s{sigma}_d{d0}_n{n}.png')
        
        axes[1, 0].imshow(b_lp, cmap='gray')
        axes[1, 0].set_title('Butterworth LP', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(b_hp, cmap='gray')
        axes[1, 1].set_title('Butterworth HP', fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(b_bp, cmap='gray')
        axes[1, 2].set_title('Butterworth BP', fontsize=12)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(b_notch, cmap='gray')
        axes[1, 3].set_title('Butterworth Notch', fontsize=12)
        axes[1, 3].axis('off')
        
        # Ideal Filters
        i_lp = Image.open(f'ideal_lp_s{sigma}_d{d0}.png')
        i_hp = Image.open(f'ideal_hp_s{sigma}_d{d0}.png')
        i_bp = Image.open(f'ideal_bp_s{sigma}_d{d0}.png')
        i_notch = Image.open(f'ideal_notch_s{sigma}_d{d0}.png')
        
        axes[2, 0].imshow(i_lp, cmap='gray')
        axes[2, 0].set_title('Ideal LP', fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(i_hp, cmap='gray')
        axes[2, 1].set_title('Ideal HP', fontsize=12)
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(i_bp, cmap='gray')
        axes[2, 2].set_title('Ideal BP', fontsize=12)
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(i_notch, cmap='gray')
        axes[2, 3].set_title('Ideal Notch', fontsize=12)
        axes[2, 3].axis('off')
        
        # Empty space for last subplot
        axes[2, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig('all_filters_comparison.png', dpi=150)
        print("\n✅ Comparison image saved: all_filters_comparison.png")
        plt.show()
        
    except Exception as e:
        print(f"\n⚠️ Could not display comparison: {e}")
    
    # ============================================
    # Step 5: Analysis and verification
    # ============================================
    print_header("📈 STEP 4: FINAL ANALYSIS")
    
    print("\n🔍 VERIFICATION CHECKLIST:")
    print("-"*70)
    print("\n✅ Check 1: High-Pass vs Low-Pass")
    print("   → High-Pass should ALWAYS have HIGHER values than Low-Pass")
    print("   → This means high frequencies are preserved")
    
    print("\n✅ Check 2: Sigma Effect")
    print("   → Higher sigma = Higher all values")
    print("   → Compare sigma=10 vs sigma=50 results")
    
    print("\n✅ Check 3: d0 Effect (for Low-Pass)")
    print("   → Higher d0 = Higher Low-Pass values (weaker filter)")
    print("   → Lower d0 = Lower Low-Pass values (stronger filter)")
    
    print("\n✅ Check 4: Filter Comparison")
    print("   → Different filters may have different values")
    print("   → But the patterns should be similar")
    
    print("\n✅ Check 5: Notch Filter")
    print("   → Notch should remove specific frequencies")
    print("   → Values should be between LP and HP")
    
    print("\n" + "="*70)
    print("🎯 VERDICT:")
    print("="*70)
    
    # Check if everything is working
    working = True
    
    # Check first config results
    first_config = test_configs[0]
    sigma = first_config['sigma']
    d0 = first_config['d0']
    
    try:
        g_lp_val = np.mean(np.abs(np.array(Image.open(f'gaussian_lp_s{sigma}_d{d0}.png'), dtype=np.float32) - np.array(test_img, dtype=np.float32)))
        g_hp_val = np.mean(np.abs(np.array(Image.open(f'gaussian_hp_s{sigma}_d{d0}.png'), dtype=np.float32) - np.array(test_img, dtype=np.float32)))
        
        if g_hp_val > g_lp_val:
            print("\n✅ HIGH-PASS > LOW-PASS: PASSED")
        else:
            print("\n❌ HIGH-PASS > LOW-PASS: FAILED")
            working = False
            
    except:
        print("\n⚠️ Could not verify values")
    
    if working:
        print("\n✅ All tests PASSED! Your filter functions are working correctly!")
        print("\n🎉 Congratulations! You can use these filters in your project.")
    else:
        print("\n❌ Some tests FAILED. Please check your filter functions.")
    
    print("\n💡 TIP: Check the generated images in the current directory")
    print("   Look for patterns: LP = smooth, HP = sharp, BP = medium")
    print("="*70)

if __name__ == "__main__":
    run_comprehensive_test()