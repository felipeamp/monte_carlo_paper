#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>

using namespace std;

#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define all(v) (v).begin(), (v).end()
#define uniq(v) (v).erase(unique(all(v)), v.end())
#define IOS ios::sync_with_stdio(0);

#define fr(a, b, c) for(int (a) = (b); (a) < (c); ++(a))
#define rp(a, b) fr(a,0,b)
#define cl(a, b) memset((a), (b), sizeof(a))
#define sc(a) scanf("%d", &a)
#define sc2(a, b) scanf("%d %d", &a, &b)
#define sc3(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define pr(a) printf("%d\n", a);
#define pr2(a, b) printf("%d %d\n", a, b)
#define pr3(a, b, c) printf("%d %d %d\n", a, b, c)
#define IOS ios::sync_with_stdio(0);
#define sz(v) (int) (v).size()

typedef unsigned long long ull;
typedef long long ll;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef pair<ll, ll> pll;

vector<unordered_map<string, string> > dict(10);

int main()
{

	//Native english speaker?
	dict[0].insert(mp("1", "English-Speaker"));
	dict[0].insert(mp("2", "Non-English-Speaker"));

	// 2 and 3 are categorical attributes
	rp(i, 27){
		string tmp = "c";
		tmp += to_string(i);
		dict[1].insert(mp(to_string(i), tmp));
		dict[2].insert(mp(to_string(i), tmp));
	}

	//summer or regular semester?
	dict[3].insert(mp("1", "Summer"));
	dict[3].insert(mp("2", "Regular"));

	// class attribute
	dict[5].insert(mp("1", "Low"));
	dict[5].insert(mp("2", "Medium"));
	dict[5].insert(mp("3", "High"));


	ifstream in; in.open("temp.csv");
	ofstream out; out.open("data.csv");
	string line;
	while(getline(in, line))
	{
		vector<string> fields;
		boost::split(fields, line, boost::is_any_of(","));
		out << dict[0][fields[0]];
		fr(i, 1, sz(fields))
		{
			if(i == 4)continue;
			out << ",";
			out << dict[i][fields[i]];
		}
		out << endl;
	}
	return 0;
}
