import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🤖 딥러닝 손실 함수와 르베그 적분 이해하기")

st.markdown("### 🎯 예측과 실제값이 주어졌을 때, 손실 함수의 평균을 어떻게 계산할까?")

# 입력값 x를 0~1에서 100개 샘플링
x = np.linspace(0, 1, 100)
true_y = np.sin(2 * np.pi * x)  # 실제값 (예시로 사인파)
pred_y = st.slider("예측값 오차 정도", 0.0, 2.0, 1.0) * true_y

# 손실 함수: MSE 사용
loss = (true_y - pred_y) ** 2

# 확률분포: uniform or 사용자 선택
distribution_type = st.selectbox("입력 데이터의 분포", ["균등분포 (Uniform)", "종모양 (Normal)"])

if distribution_type == "균등분포 (Uniform)":
    prob_density = np.ones_like(x) / len(x)
else:
    mu, sigma = 0.5, 0.15
    prob_density = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    prob_density /= prob_density.sum()  # 정규화

# 기대 손실: 르베그 적분에 해당하는 sum(loss * 확률)
expected_loss = np.sum(loss * prob_density)

# 시각화
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('x (입력)')
ax1.set_ylabel('손실 (Loss)', color=color)
ax1.plot(x, loss, color=color, label="손실 함수")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('확률 밀도', color=color)
ax2.plot(x, prob_density, color=color, linestyle='--', label="확률 분포")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
st.pyplot(fig)

st.markdown(f"### 🧮 기대 손실 (Expected Loss): {expected_loss:.4f}")
st.markdown("→ 이것이 바로 르베그 적분의 결과와 같습니다!")

st.info("르베그 적분은 손실값이 '어디에 얼마나 많이 분포되어 있는가'를 따져서 평균을 계산합니다.")
