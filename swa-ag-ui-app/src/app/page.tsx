"use client";

import { useState } from "react";
import { useCoAgent, useCopilotAction, useLangGraphInterrupt } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";

type AgentState = {
  proverbs: string[];
};

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  useCopilotAction({
    name: "setThemeColor",
    parameters: [{ name: "themeColor", description: "The theme color to set.", required: true }],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent themeColor={themeColor} />
      <CopilotSidebar
        clickOutsideToClose={false}
        defaultOpen={true}
        labels={{
          title: "Popup Assistant",
          initial:
            "👋 Hi, there! You're chatting with an agent. This agent comes with a few tools to get you started.\n\n" +
            "- **Frontend Tools**: \"Set the theme to orange\"\n" +
            "- **Shared State**: \"Write a proverb about AI\"\n" +
            "- **Generative UI**: \"Get the weather in SF\"\n\n" +
            "As you interact with the agent, you'll see the UI update in real-time to reflect the agent's **state**, **tool calls**, and **progress**.",
        }}
      />
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: {
      proverbs: ["CopilotKit may be new, but it's the best thing since sliced bread."],
    },
  });

  const [weatherLocation, setWeatherLocation] = useState<string | null>(null);
  const [showWeather, setShowWeather] = useState(false);

  useLangGraphInterrupt<{ action: string; message: string; location: string }>({
    render: ({ event, resolve }) => {
      if (event.value.action !== "confirm_weather_request") return <></>;

      return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 shadow-lg text-center">
            <p className="text-lg mb-4">{event.value.message}</p>
            <div className="flex justify-center gap-4">
              <button
                onClick={() => {
                  setWeatherLocation(event.value.location);
                  setShowWeather(true);
                  resolve("approved");
                }}
                className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600"
              >
                Approve
              </button>
              <button
                onClick={() => {
                  setShowWeather(false);
                  resolve("rejected");
                }}
                className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600"
              >
                Reject
              </button>
            </div>
          </div>
        </div>
      );
    },
  });

  useCopilotAction({
    name: "addProverb",
    parameters: [{ name: "proverb", description: "The proverb to add", required: true }],
    handler: ({ proverb }) => setState({ ...state, proverbs: [...state.proverbs, proverb] }),
  });

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen w-screen flex justify-center items-center flex-col transition-colors duration-300"
    >
      <div className="bg-white/20 backdrop-blur-md p-8 rounded-2xl shadow-xl max-w-2xl w-full">
        <h1 className="text-4xl font-bold text-white mb-2 text-center">Proverbs</h1>
        <p className="text-gray-200 text-center italic mb-6">
          This is a demonstrative page, but it could be anything you want! 🪁
        </p>

        {showWeather && weatherLocation && (
          <WeatherCard location={weatherLocation} themeColor={themeColor} />
        )}

        <hr className="border-white/20 my-6" />
        
        <div className="flex flex-col gap-3">
          {state.proverbs?.map((proverb, index) => (
            <div
              key={index}
              className="bg-white/15 p-4 rounded-xl text-white relative group hover:bg-white/20 transition-all"
            >
              <p className="pr-8">{proverb}</p>
              <button
                onClick={() =>
                  setState({
                    ...state,
                    proverbs: state.proverbs.filter((_, i) => i !== index),
                  })
                }
                className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity bg-red-500 hover:bg-red-600 text-white rounded-full h-6 w-6 flex items-center justify-center"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {state.proverbs?.length === 0 && (
          <p className="text-center text-white/80 italic my-8">
            No proverbs yet. Ask the assistant to add some!
          </p>
        )}
      </div>
    </div>
  );
}

function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-14 h-14 text-yellow-200">
      <circle cx="12" cy="12" r="5" />
      <path
        d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
        strokeWidth="2"
        stroke="currentColor"
      />
    </svg>
  );
}

function WeatherCard({ location, themeColor }: { location?: string; themeColor: string }) {
  return (
    <div style={{ backgroundColor: themeColor }} className="rounded-xl shadow-xl mt-6 mb-4 max-w-md w-full">
      <div className="bg-white/20 p-4 w-full">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white capitalize">{location}</h3>
            <p className="text-white">Current Weather</p>
          </div>
          <SunIcon />
        </div>

        <div className="mt-4 flex items-end justify-between">
          <div className="text-3xl font-bold text-white">70°</div>
          <div className="text-sm text-white">Clear skies</div>
        </div>

        <div className="mt-4 pt-4 border-t border-white">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-white text-xs">Humidity</p>
              <p className="text-white font-medium">45%</p>
            </div>
            <div>
              <p className="text-white text-xs">Wind</p>
              <p className="text-white font-medium">5 mph</p>
            </div>
            <div>
              <p className="text-white text-xs">Feels Like</p>
              <p className="text-white font-medium">72°</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}